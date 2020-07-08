from __future__ import absolute_import, division, print_function

import os
import random
import math

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)
from pytorch_pretrained_bert.tokenization import BertTokenizer

import pytorch_pretrained_zen as zen

from pytorch_pretrained_bert.crf import CRF

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_attention': False,
    'feature_flag': 'pos',
    'source': 'stanford',
}

class Attention(nn.Module):
    def __init__(self, hidden_size, word_size):
        super(Attention, self).__init__()
        self.temper = hidden_size ** 0.5
        self.word_embedding = nn.Embedding(word_size, hidden_size, padding_idx=0)

    def forward(self, word_seq, hidden_state, word_mask_matrix):
        batch_size, character_seq_len, _ = hidden_state.shape

        embedding = self.word_embedding(word_seq)

        tmp = embedding.permute(0, 2, 1)

        u = torch.matmul(hidden_state, tmp) / self.temper

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, word_mask_matrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        character_attention = torch.bmm(attention, embedding)

        return character_attention


class TwASP(nn.Module):

    def __init__(self, word2id, gram2id, feature2id, labelmap, processor, hpara, args):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec.pop('args')
        self.word2id = word2id

        self.hpara = hpara
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_size = self.hpara['max_ngram_size']
        self.use_attention = self.hpara['use_attention']

        self.gram2id = gram2id
        self.feature2id = feature2id
        self.feature_processor = processor

        if self.hpara['use_attention']:
            self.source = self.hpara['source']
            self.feature_flag = self.hpara['feature_flag']
        else:
            self.source = None
            self.feature_flag = None

        self.labelmap = labelmap
        self.num_labels = len(self.labelmap) + 1

        self.bert_tokenizer = None
        self.bert = None
        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=self.hpara['do_lower_case'])
                self.bert = BertModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                self.hpara['bert_tokenizer'] = self.bert_tokenizer
                self.hpara['config'] = self.bert.config
            else:
                self.bert_tokenizer = self.hpara['bert_tokenizer']
                self.bert = BertModel(self.hpara['config'])
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            if args.do_train:
                cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(zen.PYTORCH_PRETRAINED_BERT_CACHE),
                                                                               'distributed_{}'.format(args.local_rank))
                self.zen_tokenizer = zen.BertTokenizer.from_pretrained(args.bert_model, do_lower_case=self.hpara['do_lower_case'])
                self.zen_ngram_dict = zen.ZenNgramDict(args.bert_model, tokenizer=self.zen_tokenizer)
                self.zen = zen.modeling.ZenModel.from_pretrained(args.bert_model, cache_dir=cache_dir)
                self.hpara['zen_tokenizer'] = self.zen_tokenizer
                self.hpara['zen_ngram_dict'] = self.zen_ngram_dict
                self.hpara['config'] = self.zen.config
            else:
                self.zen_tokenizer = self.hpara['zen_tokenizer']
                self.zen_ngram_dict = self.hpara['zen_ngram_dict']
                self.zen = zen.modeling.ZenModel(self.hpapra['config'])
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.hpara['use_attention']:
            self.context_attention = Attention(hidden_size, len(self.gram2id))
            self.feature_attention = Attention(hidden_size, len(self.feature2id))
            self.classifier = nn.Linear(hidden_size * 3, self.num_labels, bias=False)
        else:
            self.context_attention = None
            self.feature_attention = None
            self.classifier = nn.Linear(hidden_size, self.num_labels, bias=False)

        self.crf = CRF(tagset_size=self.num_labels - 3, gpu=True)

        if args.do_train:
            self.spec['hpara'] = self.hpara

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_attention'] = args.use_attention
        hyper_parameters['feature_flag'] = args.feature_flag
        hyper_parameters['source'] = args.source
        return hyper_parameters

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None,
                word_seq=None, feature_seq=None, word_matrix=None, feature_matrix=None,
                input_ngram_ids=None, ngram_position_matrix=None):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        if self.context_attention is not None:
            word_attention = self.context_attention(word_seq, sequence_output, word_matrix)
            feature_attention = self.feature_attention(feature_seq, sequence_output, feature_matrix)
            conc = torch.cat([sequence_output, word_attention, feature_attention], dim=2)
        else:
            conc = sequence_output

        conc = self.dropout(conc)

        logits = self.classifier(conc)

        total_loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
        scores, tag_seq = self.crf._viterbi_decode(logits, attention_mask)

        return total_loss, tag_seq

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model, args):
        spec = spec.copy()
        res = cls(args=args, **spec)
        res.load_state_dict(model)
        return res

    def load_data(self, data_path):
        lines = readfile(data_path)

        flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]

        data = []
        if self.feature_flag is None:
            for sentence, label in lines:
                data.append((sentence, label, None, None, None, None))

        elif self.feature_flag == 'pos':
            all_feature_data = self.feature_processor.read_features(data_path, flag=flag)
            for (sentence, label), feature_list in zip(lines, all_feature_data):
                word_list = []
                syn_feature_list = []
                word_matching_position = []
                syn_matching_position = []
                for token_index, token in enumerate(feature_list):
                    current_token_pos = token['pos']
                    current_token = token['word']
                    current_feature = current_token + '_' + current_token_pos
                    if current_token not in self.gram2id:
                        current_token = '<UNK>'
                    if current_feature not in self.feature2id:
                        if current_token_pos not in self.feature2id:
                            current_feature = '<UNK>'
                        else:
                            current_feature = current_token_pos
                    word_list.append(current_token)
                    syn_feature_list.append(current_feature)

                    assert current_token in self.gram2id
                    assert current_feature in self.feature2id

                    char_index_list = token['char_index']
                    begin_char_index = max(char_index_list[0] - 2, 0)
                    end_char_index = min(char_index_list[-1] + 3, len(sentence))
                    for i in range(begin_char_index, end_char_index):
                        word_matching_position.append((i, token_index))
                        syn_matching_position.append((i, token_index))
                data.append((sentence, label, word_list, syn_feature_list,
                             word_matching_position, syn_matching_position))
        elif self.feature_flag == 'chunk':
            all_feature_data = self.feature_processor.read_features(data_path, flag=flag)
            for (sentence, label), feature_list in zip(lines, all_feature_data):
                word_list = []
                syn_feature_list = []
                word_matching_position = []
                syn_matching_position = []
                for token_index, token in enumerate(feature_list):
                    current_token_chunk_tag = token['chunk_tags'][-1]['chunk_tag']
                    assert token['chunk_tags'][-1]['height'] == 1
                    current_token = token['word']
                    current_feature = current_token + '_' + current_token_chunk_tag
                    if current_token not in self.gram2id:
                        current_token = '<UNK>'
                    if current_feature not in self.feature2id:
                        if current_token_chunk_tag not in self.feature2id:
                            current_feature = '<UNK>'
                        else:
                            current_feature = current_token_chunk_tag
                    word_list.append(current_token)
                    syn_feature_list.append(current_feature)

                    assert current_token in self.gram2id
                    assert current_feature in self.feature2id

                    token_index_list = token['chunk_tags'][-1]['range']
                    char_index_list = token['char_index']

                    for i in char_index_list:
                        for j in token_index_list:
                            word_matching_position.append((i, j))
                            syn_matching_position.append((i, j))
                    word_matching_position = list(set(word_matching_position))
                    syn_matching_position = list(set(syn_matching_position))
                data.append((sentence, label, word_list, syn_feature_list,
                             word_matching_position, syn_matching_position))
        elif self.feature_flag == 'dep':
            all_feature_data = self.feature_processor.read_features(data_path, flag=flag)
            for (sentence, label), feature_list in zip(lines, all_feature_data):
                word_list = []
                syn_feature_list = []
                word_matching_position = []
                syn_matching_position = []
                for token_index, token in enumerate(feature_list):
                    current_token_dep_tag = token['dep']
                    current_token = token['word']
                    current_feature = current_token + '_' + current_token_dep_tag
                    if current_token not in self.gram2id:
                        current_token = '<UNK>'
                    if current_feature not in self.feature2id:
                        if current_token_dep_tag not in self.feature2id:
                            current_feature = '<UNK>'
                        else:
                            current_feature = current_token_dep_tag
                    word_list.append(current_token)
                    syn_feature_list.append(current_feature)

                    assert current_token in self.gram2id
                    assert current_feature in self.feature2id

                    if token['governed_index'] < 0:
                        token_index_list = [token_index]
                        char_index_list = token['char_index']
                    else:
                        governed_index = token['governed_index']
                        token_index_list = [token_index, governed_index]
                        governed_token = feature_list[governed_index]
                        char_index_list = token['char_index'] + governed_token['char_index']

                    for i in char_index_list:
                        for j in token_index_list:
                            word_matching_position.append((i, j))
                            syn_matching_position.append((i, j))
                    word_matching_position = list(set(word_matching_position))
                    syn_matching_position = list(set(syn_matching_position))
                data.append((sentence, label, word_list, syn_feature_list,
                             word_matching_position, syn_matching_position))
        else:
            raise ValueError()

        examples = []
        for i, (
        sentence, label, word_list, syn_feature_list, word_matching_position, syn_matching_position) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = ' '.join(sentence)
            text_b = None
            if word_list is not None:
                word = ' '.join(word_list)
                word_list_len = len(word_list)
            else:
                word = None
                word_list_len = 0
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word,
                             syn_feature=syn_feature_list, word_matrix=word_matching_position,
                             syn_matrix=syn_matching_position, sent_len=len(sentence), word_list_len=word_list_len))
        return examples

    def convert_examples_to_features(self, examples):

        max_seq_length = min(int(max([e.sent_len for e in examples]) * 1.1 + 2), self.max_seq_length)

        if self.use_attention:
            max_ngram_size = max(min(max([e.word_list_len for e in examples]), self.max_ngram_size), 1)

        features = []
        tokenizer = self.bert_tokenizer if self.bert_tokenizer is not None else self.zen_tokenizer

        for (ex_index, example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []

            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)

            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    if labels[i] in self.labelmap:
                        label_ids.append(self.labelmap[labels[i]])
                    else:
                        label_ids.append(self.labelmap['<UNK>'])
            ntokens.append("[SEP]")

            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            if self.use_attention:
                wordlist = example.word
                wordlist = wordlist.split(' ') if len(wordlist) > 0 else []
                syn_features = example.syn_feature
                word_matching_position = example.word_matrix
                syn_matching_position = example.syn_matrix
                word_ids = []
                feature_ids = []
                word_matching_matrix = np.zeros((max_seq_length, max_ngram_size), dtype=np.int)
                syn_matching_matrix = np.zeros((max_seq_length, max_ngram_size), dtype=np.int)

                if len(wordlist) > max_ngram_size:
                    wordlist = wordlist[:max_ngram_size]
                    syn_features = syn_features[:max_ngram_size]

                for word in wordlist:
                    if word == '':
                        continue
                    try:
                        word_ids.append(self.gram2id[word])
                    except KeyError:
                        print(word)
                        print(wordlist)
                        print(textlist)
                        raise KeyError()
                for feature in syn_features:
                    feature_ids.append(self.feature2id[feature])

                while len(word_ids) < max_ngram_size:
                    word_ids.append(0)
                    feature_ids.append(0)
                for position in word_matching_position:
                    char_p = position[0] + 1
                    word_p = position[1]
                    if char_p > max_seq_length - 2 or word_p > max_ngram_size - 1:
                        continue
                    else:
                        word_matching_matrix[char_p][word_p] = 1

                for position in syn_matching_position:
                    char_p = position[0] + 1
                    word_p = position[1]
                    if char_p > max_seq_length - 2 or word_p > max_ngram_size - 1:
                        continue
                    else:
                        syn_matching_matrix[char_p][word_p] = 1

                assert len(word_ids) == max_ngram_size
                assert len(feature_ids) == max_ngram_size
            else:
                word_ids = None
                feature_ids = None
                word_matching_matrix = None
                syn_matching_matrix = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                random.shuffle(ngram_matches)

                max_ngram_in_seq_proportion = math.ceil(
                    (len(tokens) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                  dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              word_ids=word_ids,
                              syn_feature_ids=feature_ids,
                              word_matching_matrix=word_matching_matrix,
                              syn_matching_matrix=syn_matching_matrix,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
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
        if self.hpara['use_attention']:
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
        if self.hpara['use_zen']:
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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None,
                 syn_feature=None, word_matrix=None, syn_matrix=None, sent_len=None, word_list_len=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.word_matrix = word_matrix
        self.syn_matrix = syn_matrix
        self.syn_feature = syn_feature
        self.sent_len = sent_len
        self.word_list_len = word_list_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, syn_feature_ids=None, word_matching_matrix=None, syn_matching_matrix=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.syn_feature_ids = syn_feature_ids
        self.word_matching_matrix = word_matching_matrix
        self.syn_matching_matrix = syn_matching_matrix

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split('\t')
        char = splits[0]
        l = splits[-1][:-1]

        sentence.append(char)
        label.append(l)

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data

