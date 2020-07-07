# TwASP

This is the implementation of [Joint Chinese Word Segmentation and Part-of-speech Tagging via Two-way Attentions of Auto-analyzed Knowledge](https://www.aclweb.org/anthology/2020.acl-main.735/) at ACL2020.

We will keep updating this repository these days.

## Requirements

Our code works with the following environment.
* `python=3.6`
* `pytorch=1.1`

## Downloading BERT and ZEN

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model form [here](https://github.com/sinovation/ZEN).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Datasets

We use [CTB5](https://catalog.ldc.upenn.edu/LDC2005T01), [CTB6](https://catalog.ldc.upenn.edu/LDC2007T36), [CTB7](https://catalog.ldc.upenn.edu/LDC2010T07), [CTB9](https://catalog.ldc.upenn.edu/LDC2016T13), and [Universal Dependencies 2.4](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2988) (UD) in our paper.

To obtain and pre-process the data, please go to `data_preprocessing` directory and run `getdata.sh` under that directory. This script will download and process the official data from UD. For CTB5 (LDC05T01), CTB6 (LDC07T36), CTB7 (LDC10T07), and CTB9 (LDC2016T13), you need to obtain the official data yourself, and then put the raw data directory under the `data_preprocessing` directory.

The script will also download the [Stanford CoreNLP Toolkit v3.9.2](https://stanfordnlp.github.io/CoreNLP/history.html) (SCT) and [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) (BNP) to obtain the auto-analyzed syntactic knowledge. You can refer to their cite for more information.

All processed data will appear in `data` directory organized by the datasets, where each of them contains the files with the same file names under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test model on a specific dataset with the part-of-speech (POS) knowledge from Stanford CoreNLP Toolkit v3.9.2 (SCT) in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model
* `--do_test`: test the model
* `--use_bert`: use BERT as encoder
* `--use_zen`: use ZEN as encoder
* `--bert_model`: the directory of pre-trained BERT/ZEN model
* `--use_attention`: use two-way attention
* `--source`: the toolkit to be use (`stanford` or `berkeley`)
* `--feature_flag`: use `pos`, `chunk`, or `dep` knowledge
* `--model_name`: the name of model to save 


