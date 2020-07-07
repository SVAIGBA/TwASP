from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
from os import path

import numpy as np
import torch

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from twasp_helper import get_word2id, getlabels, request_features_from_stanford, request_features_from_berkeley, \
    berkeley_feature_processor, stanford_feature_processor, get_feature2id
from twasp_eval import eval_sentence, pos_evaluate_word_PRF, pos_evaluate_OOV
from twasp_model import TwASP
import datetime


def train(args):

    if args.use_bert and args.use_zen:
        raise ValueError('We cannot use both BERT and ZEN')

    if not os.path.exists('./logs/'):
        os.mkdir('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    else:
        output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)

    label_map = getlabels(args.train_data_path)
    id2label = {v: k for k, v in label_map.items()}
    id2label[0] = 'O'
    word2id = get_word2id(args.train_data_path)

    if args.use_attention:
        if args.source == 'stanford':
            request_features_from_stanford(args.train_data_path, flag='train')
            request_features_from_stanford(args.eval_data_path, flag='test')
            processor = stanford_feature_processor()
        elif args.source == 'berkeley':
            request_features_from_berkeley(args.train_data_path, flag='train')
            request_features_from_berkeley(args.eval_data_path, flag='test')
            processor = berkeley_feature_processor()
        else:
            raise ValueError('Source must be one of \'stanford\' or \'berkeley\' if attentions are used.')
        gram2id, feature2id = get_feature2id(args.train_data_path, processor, args.feature_flag, args.feature_threshold)
    else:
        processor = None
        gram2id = None
        feature2id = None

    joint_model = TwASP(word2id, gram2id, feature2id, label_map, processor, args)

    train_examples = joint_model.load_data(args.train_data_path, flag='train')
    eval_examples = joint_model.load_data(args.eval_data_path, flag='test')
    num_labels = joint_model.num_labels
    convert_examples_to_features = joint_model.convert_examples_to_features

    total_params = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        joint_model.half()
    joint_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        joint_model = DDP(joint_model)
    elif n_gpu > 1:
        joint_model = torch.nn.DataParallel(joint_model)

    param_optimizer = list(joint_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0

    best_epoch = -1
    best_wp = -1
    best_wr = -1
    best_wf = -1
    best_woov = -1
    best_pp = -1
    best_pr = -1
    best_pf = -1
    best_poov = -1
    history = {'epoch': [], 'word_p': [], 'word_r': [], 'word_f': [], 'word_oov': [],
               'pos_p': [], 'pos_r': [], 'pos_f': [], 'pos_oov': []}
    num_of_no_improvement = 0
    patient = args.patient

    if args.do_train:

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            np.random.shuffle(train_examples)
            joint_model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
                batch_examples = train_examples[start_index: min(start_index +
                                                                 args.train_batch_size, len(train_examples))]
                if len(batch_examples) == 0:
                    continue
                train_features = convert_examples_to_features(batch_examples)
                feature_ids, input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, \
                segment_ids, valid_ids, word_ids, word_matching_matrix = feature2input(args, device, train_features)

                loss, _ = joint_model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, word_ids,
                                      feature_ids, word_matching_matrix, word_matching_matrix, ngram_ids, ngram_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            joint_model.to(device)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                joint_model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true = []
                y_pred = []
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]
                    eval_features = convert_examples_to_features(eval_batch_examples)

                    feature_ids, input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids, word_ids, word_matching_matrix = feature2input(args, device, eval_features)

                    with torch.no_grad():
                        _, tag_seq = joint_model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                                 word_ids, feature_ids, word_matching_matrix, word_matching_matrix,
                                                 ngram_ids, ngram_positions)

                    # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
                    # logits = logits.detach().cpu().numpy()
                    logits = tag_seq.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1:
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(id2label[label_ids[i][j]])
                                temp_2.append(id2label[logits[i][j]])

                y_true_all = []
                y_pred_all = []
                sentence_all = []
                for y_true_item in y_true:
                    y_true_all += y_true_item
                for y_pred_item in y_pred:
                    y_pred_all += y_pred_item
                for example, y_true_item in zip(eval_examples, y_true):
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    if len(y_true_item) != len(sen):
                        print(len(sen))
                        sen = sen[:len(y_true_item)]
                    sentence_all.append(sen)
                (wp, wr, wf), (pp, pr, pf) = pos_evaluate_word_PRF(y_pred_all, y_true_all)
                woov, poov = pos_evaluate_OOV(y_pred, y_true, sentence_all, word2id)
                history['epoch'].append(epoch)
                history['word_p'].append(wp)
                history['word_r'].append(wr)
                history['word_f'].append(wf)
                history['word_oov'].append(woov)
                history['pos_p'].append(pp)
                history['pos_r'].append(pr)
                history['pos_f'].append(pf)
                history['pos_oov'].append(poov)
                logger.info("=======entity level========")
                logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
                            epoch + 1, wp, wr, wf, woov)
                logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
                            epoch + 1, pp, pr, pf, poov)
                logger.info("=======entity level========")
                # the evaluation method of NER
                report = classification_report(y_true, y_pred, digits=4)

                if args.model_name is not None:
                    if not os.path.exists(output_model_dir):
                        os.mkdir(output_model_dir)

                if pf > best_pf:
                    best_epoch = epoch + 1
                    best_wp = wp
                    best_wr = wr
                    best_wf = wf
                    best_woov = woov
                    best_pp = pp
                    best_pr = pr
                    best_pf = pf
                    best_poov = poov
                    num_of_no_improvement = 0

                    if args.model_name:
                        output_model_dir = path.join('./models', args.model_name + '_' + now_time)
                        if not os.path.exists(output_model_dir):
                            os.mkdir(output_model_dir)

                        with open(os.path.join(output_model_dir, 'POS_result.txt'), "w") as writer:
                            writer.write("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f" %
                                         (epoch + 1, wp, wr, wf, woov))
                            writer.write("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f" %
                                         (epoch + 1, pp, pr, pf, poov))
                            for i in range(len(y_pred)):
                                sentence = eval_examples[i].text_a
                                seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
                                writer.write('True: %s\n' % seg_true_str)
                                writer.write('Pred: %s\n\n' % seg_pred_str)

                        best_eval_model_path = os.path.join(output_model_dir, 'model.pt')

                        if n_gpu > 1:
                            torch.save({
                                'spec': joint_model.module.spec,
                                'state_dict': joint_model.module.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                        else:
                            torch.save({
                                'spec': joint_model.spec,
                                'state_dict': joint_model.state_dict(),
                                # 'trainer': optimizer.state_dict(),
                            }, best_eval_model_path)
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
                break

        logger.info("\n=======best f entity level========")
        logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
                    best_epoch, best_wp, best_wr, best_wf, best_woov)
        logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
                    best_epoch, best_pp, best_pr, best_pf, best_poov)
        logger.info("\n=======best f entity level========")

        if args.model_name is not None:
            with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
                json.dump(history, f)
                f.write('\n')


def feature2input(args, device, feature):
    all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)

    input_ids = all_input_ids.to(device)
    input_mask = all_input_mask.to(device)
    segment_ids = all_segment_ids.to(device)
    label_ids = all_label_ids.to(device)
    valid_ids = all_valid_ids.to(device)
    l_mask = all_lmask_ids.to(device)
    if args.use_attention:
        all_word_ids = torch.tensor([f.word_ids for f in feature], dtype=torch.long)
        all_feature_ids = torch.tensor([f.syn_feature_ids for f in feature], dtype=torch.long)
        all_word_matching_matrix = torch.tensor([f.word_matching_matrix for f in feature],
                                                dtype=torch.float)

        word_ids = all_word_ids.to(device)
        feature_ids = all_feature_ids.to(device)
        word_matching_matrix = all_word_matching_matrix.to(device)
    else:
        word_ids = None
        feature_ids = None
        word_matching_matrix = None
    if args.use_zen:
        all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
        all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
        # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
        # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
        # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

        ngram_ids = all_ngram_ids.to(device)
        ngram_positions = all_ngram_positions.to(device)
    else:
        ngram_ids = None
        ngram_positions = None
    return feature_ids, input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, segment_ids, valid_ids, word_ids, word_matching_matrix


def test(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    joint_model_checkpoint = torch.load(args.eval_model)
    joint_model = TwASP.from_spec(joint_model_checkpoint['spec'], joint_model_checkpoint['state_dict'])

    if joint_model.use_attention:
        if joint_model.spec['args'].source == 'stanford':
            request_features_from_stanford(args.eval_data_path, flag='test')
        elif joint_model.spec['args'].source == 'berkeley':
            request_features_from_berkeley(args.eval_data_path, flag='test')
        else:
            raise ValueError('Source must be one of \'stanford\' or \'berkeley\' if attentions are used.')

    eval_examples = joint_model.load_data(args.eval_data_path, flag='test')
    convert_examples_to_features = joint_model.convert_examples_to_features
    num_labels = joint_model.num_labels
    word2id = joint_model.word2id
    model_args = joint_model.spec['args']
    label_map = {v: k for k, v in joint_model.labelmap.items()}
    label_map[0] = 'O'

    if args.fp16:
        joint_model.half()
    joint_model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        joint_model = DDP(joint_model)
    elif n_gpu > 1:
        joint_model = torch.nn.DataParallel(joint_model)

    joint_model.to(device)

    joint_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        feature_ids, input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, \
        segment_ids, valid_ids, word_ids, word_matching_matrix = feature2input(model_args, device, eval_features)

        with torch.no_grad():
            _, tag_seq = joint_model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                                     word_ids, feature_ids, word_matching_matrix, word_matching_matrix,
                                     ngram_ids, ngram_positions)

        # logits = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
        # logits = logits.detach().cpu().numpy()
        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

    y_true_all = []
    y_pred_all = []
    sentence_all = []
    for y_true_item in y_true:
        y_true_all += y_true_item
    for y_pred_item in y_pred:
        y_pred_all += y_pred_item
    for example, y_true_item in zip(eval_examples, y_true):
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        if len(y_true_item) != len(sen):
            print(len(sen))
            sen = sen[:len(y_true_item)]
        sentence_all.append(sen)
    (wp, wr, wf), (pp, pr, pf) = pos_evaluate_word_PRF(y_pred_all, y_true_all)
    woov, poov = pos_evaluate_OOV(y_pred, y_true, sentence_all, word2id)

    print(args.eval_data_path)
    print('\n')
    print("word P: %f, word R: %f, word F: %f, word OOV: %f" % (wp, wr, wf, woov))
    print("pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f" % (pp, pr, pf, poov))


def predict(args):

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument("--use_attention",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--source', type=str, default=None, help="")
    parser.add_argument('--feature_flag', type=str, default=None, help="")
    parser.add_argument('--feature_threshold', type=int, default=1, help="")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
