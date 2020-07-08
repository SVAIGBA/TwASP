import argparse
import re
from tqdm import tqdm
import os
from os import path
from collections import defaultdict
from corenlp import StanfordCoreNLP
from nltk.tree import Tree
import json
import copy
import benepar
import nltk

FULL_MODEL = './data_preprocessing/stanford-corenlp-full-2018-10-05'

# The 12 labels follows https://www.aclweb.org/anthology/P06-2013/
chunk_pos = ['ADJP', 'ADVP', 'CLP', 'DNP', 'DP', 'DVP', 'LCP', 'LST', 'NP', 'PP', 'QP', 'VP']


def read_tsv(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = re.split('\\s+', line)
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

    return sentence_list, label_list


def get_word2id(train_path):
    word2id = {'<PAD>': 0}
    word = ''
    index = 1
    for line in open(train_path):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][0]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id


def merge_results(results):
    merged = {'index': 0, 'parse': '', 'basicDependencies': [], 'tokens': []}

    # merge fix token
    token_index = 1
    token_start_index = [0]
    for i, result in enumerate(results):
        tokens = result['tokens']
        for token in tokens:
            copy_token = copy.deepcopy(token)
            copy_token['index'] = token_index
            token_index += 1
            merged['tokens'].append(copy_token)
        token_start_index.append(token_index-1)

    # merge parse
    new_parse_str = '(ROOT '
    for result in results:
        parse = result['parse']
        new_parse_str += parse
        new_parse_str += ' '
    new_parse_str += ')'
    merged['parse'] = new_parse_str
    Tree.fromstring(new_parse_str)

    for i, result in enumerate(results):
        dep_list = result['basicDependencies']
        for dep in dep_list:
            copy_dep = copy.deepcopy(dep)
            if not copy_dep['governor'] == 0:
                copy_dep['governor'] += token_start_index[i]
            copy_dep['dependent'] += token_start_index[i]
            merged['basicDependencies'].append(copy_dep)
    return merged


def request_features_from_stanford(data_path):
    data_dir = data_path[:data_path.rfind('/')]
    flag = data_path[data_path.rfind('/') + 1: data_path.rfind('.')]

    if os.path.exists(path.join(data_dir, flag + '.stanford.json')):
        print('The Stanford data file for %s already exists!' % str(data_path))
        return None

    print('Requesting Stanford results for %s' % str(data_path))

    all_sentences, _ = read_tsv(data_path)
    sentences_str = []
    for sentence in all_sentences:
        sentences_str.append(''.join(sentence))

    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='zh') as nlp:
        for sentence in tqdm(sentences_str):
            results = nlp.request(annotators='parse,depparse', data=sentence)
            # result = results['sentences'][0]
            result = merge_results(results['sentences'])
            all_data.append(result)
    # assert len(all_data) == len(sentences_str)
    with open(path.join(data_dir, flag + '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


def request_features_from_berkeley(data_path):
    data_dir = data_path[:data_path.rfind('/')]
    flag = data_path[data_path.rfind('/') + 1: data_path.rfind('.')]

    if not os.path.exists(path.join(data_dir, flag + '.stanford.json')):
        print('Do not find the Stanford data file\nRequesting Stanford segmentation results for %s' % str(data_path))
        request_features_from_stanford(data_path, flag)
    else:
        print('The Stanford data file for %s already exists!' % str(data_path))
    if os.path.exists(path.join(data_dir, flag + '.berkeley.json')):
        print('The Berkeley data file for %s already exists!' % str(data_path))
        return None

    print('Requesting Berkeley results for %s' % str(data_path))

    berkeley_parser = benepar.Parser("benepar_zh")

    print('processing: ', flag)
    all_data = read_json(path.join(data_dir, flag + '.stanford.json'))
    berkeley_all_data = []

    for data in tqdm(all_data):
        berkeley_data = {}
        tokens = data['tokens']
        berkeley_data['tokens'] = copy.deepcopy(tokens)
        word_list = [token['word'] for token in tokens]
        parse = berkeley_parser.parse(word_list)
        str_parse = str(parse)
        parse_tree = Tree.fromstring(str_parse)

        for i, s in enumerate(parse_tree.subtrees(lambda t: t.height() == 2)):
            if not s[0] == word_list[i]:
                s[0] = word_list[i]

        berkeley_data['parse'] = str(parse_tree)
        pos_tags = parse_tree.pos()

        for i, (bt, (w, pos)) in enumerate(zip(berkeley_data['tokens'], pos_tags)):
            # w = w_pos[0]
            # pos = w_pos[1]
            # try:
            assert bt['word'] == w
            # except AssertionError:
            #     print('error in sentence: %s' % ''.join(word_list))
            #     print('word error: excepted %s, get %s' % (bt['word'], w))
            # else:
            berkeley_data['tokens'][i]['pos'] = pos
        berkeley_all_data.append(berkeley_data)

    del berkeley_parser

    with open(path.join(data_dir, flag + '.berkeley.json'), 'w', encoding='utf8') as f:
        for berkeley_data in berkeley_all_data:
            json.dump(berkeley_data, f, ensure_ascii=False)
            f.write('\n')


def get_feature2id(data_path, feature_processor, feature_flag, min_threshold=1):
    all_feature2count = feature_processor.read_feature2count(data_path)
    gram2count = all_feature2count['gram2count']
    if feature_flag == 'pos':
        feature2count = all_feature2count['pos_tag2count']
    elif feature_flag == 'chunk':
        feature2count = all_feature2count['chunk_tag2count']
    elif feature_flag == 'dep':
        feature2count = all_feature2count['dep_tag2count']
    else:
        raise ValueError()
    gram2id = {'<PAD>': 0, '<UNK>': 1}
    feature2id = {'<PAD>': 0, '<UNK>': 1}
    gram_index = 2
    feature_index = 2
    for gram, count in gram2count.items():
        if count > min_threshold:
            gram2id[gram] = gram_index
            gram_index += 1
    for feature, count in feature2count.items():
        if count > min_threshold:
            feature2id[feature] = feature_index
            feature_index += 1
    return gram2id, feature2id


def getlabels(train_path):
    _, all_labels = read_tsv(train_path)

    label2id = {'<UNK>': 1, 'O': 2}
    index = 3
    for label_list in all_labels:
        for label in label_list:
            if label not in label2id:
                label2id[label] = index
                index += 1
    label2id['[CLS]'] = index
    index += 1
    label2id['[SEP]'] = index
    return label2id


def read_json(data_path):
    data = []
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            data.append(json.loads(line))
    return data


class berkeley_feature_processor:

    def change_tree(self, word_list, t, index):
        for i, subtree in enumerate(t):
            if type(subtree) == nltk.tree.Tree:
                self.change_tree(word_list, subtree, index)
            elif type(subtree) == tuple:
                newVal = (subtree[0], word_list[index])
                subtree = newVal
                t[i] = subtree

    def read_feature2count(self, data_path):
        data_dir = data_path[:data_path.rfind('/')]
        all_data = read_json(path.join(data_dir, 'train.berkeley.json'))
        gram2count = defaultdict(int)
        pos_tag2count = defaultdict(int)
        chunk_tag2count = defaultdict(int)

        for data in all_data:
            tokens = data['tokens']
            for token in tokens:
                gram2count[token['word']] += 1
                pos_tag2count[token['pos']] += 1
                pos_tag2count[token['word'] + '_' + token['pos']] += 1

            coparse = Tree.fromstring(data['parse'])
            for s in coparse.subtrees(lambda t: t.label() in chunk_pos):
                leaves = s.leaves()
                node = s.label()
                chunk_tag2count[node] += 1
                for leaf in leaves:
                    chunk_tag2count[leaf + '_' + node] += 1
            chunk_tag2count['ROOT'] = 100

        feature2count = {'gram2count': gram2count, 'pos_tag2count': pos_tag2count,
                         'chunk_tag2count': chunk_tag2count}
        return feature2count

    def read_features(self, data_path, flag):
        data_dir = data_path[:data_path.rfind('/')]
        all_data = read_json(path.join(data_dir, flag + '.berkeley.json'))
        all_feature_data = []
        for data in all_data:
            sentence_feature = []
            sentence = ''
            words = []
            tokens = data['tokens']
            for token in tokens:
                feature_dict = {}
                feature_dict['word'] = token['word']
                words.append(token['word'])
                sentence += token['word']
                start_index = token['characterOffsetBegin']
                end_index = token['characterOffsetEnd']
                feature_dict['char_index'] = [i for i in range(start_index, end_index)]
                feature_dict['pos'] = token['pos']
                sentence_feature.append(feature_dict)

            c_parse = Tree.fromstring(data['parse'])
            current_index = 0
            for s in c_parse.subtrees(lambda t: t.label() in chunk_pos):
                leaves = s.leaves()
                if len(leaves) == 0:
                    continue
                node = s.label()
                index = words[current_index:].index(leaves[0]) + current_index
                current_index = index
                for i, leaf in enumerate(leaves):
                    if 'chunk_tags' not in sentence_feature[index + i]:
                        sentence_feature[index + i]['chunk_tags'] = []
                    sentence_feature[index + i]['chunk_tags'].append({'chunk_tag': node, 'height': 0,
                                                                      'range': (index, index + len(leaves))})
                    for chunk_tag in sentence_feature[index + i]['chunk_tags']:
                        chunk_tag['height'] += 1
            for token in sentence_feature:
                if 'chunk_tags' not in token:
                    token['chunk_tags'] = [{'chunk_tag': 'ROOT', 'height': 1, 'range': (0, len(sentence_feature))}]

            all_feature_data.append(sentence_feature)
        return all_feature_data


class stanford_feature_processor:

    def read_feature2count(self, data_path):
        data_dir = data_path[:data_path.rfind('/')]
        all_data = read_json(path.join(data_dir, 'train.stanford.json'))
        gram2count = defaultdict(int)
        pos_tag2count = defaultdict(int)
        chunk_tag2count = defaultdict(int)
        dep_tag2count = defaultdict(int)

        for data in all_data:
            tokens = data['tokens']
            for token in tokens:
                gram2count[token['word']] += 1
                pos_tag2count[token['pos']] += 1
                pos_tag2count[token['word'] + '_' + token['pos']] += 1
            deparse = data['basicDependencies']
            for word in deparse:
                dep_tag2count[word['dep']] += 1
                dep_tag2count[word['dependentGloss'] + '_' + word['dep']] += 1

            coparse = Tree.fromstring(data['parse'])
            for s in coparse.subtrees(lambda t: t.label() in chunk_pos):
                leaves = s.leaves()
                node = s.label()
                chunk_tag2count[node] += 1
                for leaf in leaves:
                    chunk_tag2count[leaf + '_' + node] += 1
            chunk_tag2count['ROOT'] = 100

        feature2count = {'gram2count': gram2count, 'pos_tag2count': pos_tag2count,
                         'chunk_tag2count': chunk_tag2count, 'dep_tag2count': dep_tag2count}
        return feature2count

    def read_features(self, data_path, flag):
        data_dir = data_path[:data_path.rfind('/')]
        all_data = read_json(path.join(data_dir, flag + '.stanford.json'))
        all_feature_data = []
        for data in all_data:
            sentence_feature = []
            sentence = ''
            words = []
            tokens = data['tokens']
            for token in tokens:
                feature_dict = {}
                feature_dict['word'] = token['word']
                words.append(token['word'])
                sentence += token['word']
                start_index = token['characterOffsetBegin']
                end_index = token['characterOffsetEnd']
                feature_dict['char_index'] = [i for i in range(start_index, end_index)]
                feature_dict['pos'] = token['pos']
                sentence_feature.append(feature_dict)

            deparse = data['basicDependencies']
            for dep in deparse:
                dependent_index = dep['dependent'] - 1
                sentence_feature[dependent_index]['dep'] = dep['dep']
                sentence_feature[dependent_index]['governed_index'] = dep['governor'] - 1

            c_parse = Tree.fromstring(data['parse'])
            current_index = 0
            for s in c_parse.subtrees(lambda t: t.label() in chunk_pos):
                leaves = s.leaves()
                if len(leaves) == 0:
                    continue
                node = s.label()
                index = words[current_index:].index(leaves[0]) + current_index
                current_index = index
                for i, leaf in enumerate(leaves):
                    if 'chunk_tags' not in sentence_feature[index + i]:
                        sentence_feature[index + i]['chunk_tags'] = []
                    sentence_feature[index + i]['chunk_tags'].append({'chunk_tag': node, 'height': 0,
                                                                      'range': (index, index + len(leaves))})
                    for chunk_tag in sentence_feature[index + i]['chunk_tags']:
                        chunk_tag['height'] += 1
            for token in sentence_feature:
                if 'chunk_tags' not in token:
                    token['chunk_tags'] = [{'chunk_tag': 'ROOT', 'height': 1, 'range': (0, len(sentence_feature))}]

            all_feature_data.append(sentence_feature)
        return all_feature_data


def extract_ngram(all_sentences, min_feq=0, ngram_len=10):
    n_gram_dict = {}

    new_all_sentences = []

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, ngram_len+1):
                if i + n > len(sentence):
                    break
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                else:
                    n_gram_dict[n_gram] += 1
    new_ngram_dict = {gram: c for gram, c in n_gram_dict.items() if c > min_feq}
    return new_ngram_dict


def renew_ngram_by_freq(all_sentences, ngram2count, min_feq, ngram_len=10):
    new_ngram2count = {}

    new_all_sentences = []

    for sen in all_sentences:
        str_sen = ''.join(sen)
        new_sen = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', str_sen)
        for s in new_sen:
            if len(s) > 0:
                new_all_sentences.append(s)

    for sentence in new_all_sentences:
        for i in range(len(sentence)):
            for n in range(1, ngram_len+1):
                if i + n > len(sentence):
                    break
                n_gram = ''.join(sentence[i: i + n])
                if n_gram not in ngram2count:
                    continue
                if n_gram not in new_ngram2count:
                    new_ngram2count[n_gram] = 1
                else:
                    new_ngram2count[n_gram] += 1
    new_ngram_dict = {gram: c for gram, c in new_ngram2count.items() if c > min_feq}
    return new_ngram_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    args = parser.parse_args()
    base_min_freq = 1
    av_threshold = 2

    min_freq = base_min_freq

    print('min freq: %d' % min_freq)

    data_dir = path.join(DATA_DIR, args.dataset)

    print(data_dir)

    # getlabels(data_dir)

    # get_word2id(data_dir)

    # be(data_dir, 0, 10)

    # oov_stat(data_dir, 'train')
    # oov_stat(data_dir, 'dev')
    # oov_stat(data_dir, 'test')
    # request_features_from_stanford(data_dir, 'train')
    # request_features_from_stanford(data_dir, 'dev')
    # request_features_from_stanford(data_dir, 'test')

    # request_features_from_stanford(data_dir, 'bc')
    # request_features_from_stanford(data_dir, 'bn')
    # request_features_from_stanford(data_dir, 'cs')
    # request_features_from_stanford(data_dir, 'df')
    # request_features_from_stanford(data_dir, 'mz')
    # request_features_from_stanford(data_dir, 'nw')
    # request_features_from_stanford(data_dir, 'sc')
    # request_features_from_stanford(data_dir, 'wb')

    # request_features_from_stanford('./data/POS/demo', 'demo')

    # sfp = stanford_feature_processor(data_dir)
    # sfp._pre_processing()
    # sfp.read_features('train')
    # sfp.read_features('test')
    # sfp.feature_stat()

    # bek = berkeley_feature_processor(data_dir)
    # bek.request_knoledge('train')
    # bek.request_knoledge('dev')
    # bek.request_knoledge('test')
    # bek.request_knoledge('demo')
    # bek._pre_processing()
    # bek.feature_stat()

    # attentionn_gram_stat(data_dir, 0, 10)

    print('')

    # exit()

