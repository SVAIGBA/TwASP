# TwASP

This is the implementation of [Joint Chinese Word Segmentation and Part-of-speech Tagging via Two-way Attentions of Auto-analyzed Knowledge](https://www.aclweb.org/anthology/2020.acl-main.735/) at ACL2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `chenguimin@chuangxin.com`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at ACL2020.

```
@inproceedings{tian-etal-2020-joint,
    title = "Joint Chinese Word Segmentation and Part-of-speech Tagging via Two-way Attentions of Auto-analyzed Knowledge",
    author = "Tian, Yuanhe and Song, Yan and Ao, Xiang and Xia, Fei and Quan, Xiaojun and Zhang, Tong and Wang, Yonggang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    pages = "8286--8296",
}
```

## Requirements

Our code works with the following environment.
* `python=3.6`
* `pytorch=1.1`

To run [Stanford CoreNLP Toolkit](https://stanfordnlp.github.io/CoreNLP/cmdline.html), you need 
* `Java 8`

To run [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser), you need
* `tensorfolw==1.13.1`
* `benepar[cpu]`
* `cython`

Note that Berkeley Neural Parser does not support `TensorFlow 2.0`.

You can refer to their websites for more information.

## Downloading BERT, ZEN and TwASP

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model form [here](https://github.com/sinovation/ZEN).

For TwASP, you can download the models we trained in our experiments from [here](https://github.com/SVAIGBA/TwASP/tree/master/models).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` folder.

## Datasets

We use [CTB5](https://catalog.ldc.upenn.edu/LDC2005T01), [CTB6](https://catalog.ldc.upenn.edu/LDC2007T36), [CTB7](https://catalog.ldc.upenn.edu/LDC2010T07), [CTB9](https://catalog.ldc.upenn.edu/LDC2016T13), and [Universal Dependencies 2.4](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2988) (UD) in our paper.

To obtain and pre-process the data, you can go to `data_preprocessing` directory and run `getdata.sh`. This script will download and process the official data from UD. For CTB5 (LDC05T01), CTB6 (LDC07T36), CTB7 (LDC10T07), and CTB9 (LDC2016T13), you need to obtain the official data yourself, and then put the raw data folder under the `data_preprocessing` directory.

The script will also download the [Stanford CoreNLP Toolkit v3.9.2](https://stanfordnlp.github.io/CoreNLP/history.html) (SCT) and [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) (BNP) to obtain the auto-analyzed syntactic knowledge. You can refer to their website for more information.

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

## Predicting

`run_sample.sh` contains the command line to segment and tag the sentences in an input file ([./sample_data/sentence.txt](./sample_data/sentence.txt)).

Here are some important parameters:

* `--do_predict`: segment and tag the sentences using a pre-trained TwASP model.
* `--input_file`: the file contains sentences to be segmented and tagged. Each line contains one sentence; you can refer to [a sample input file](./sample_data/sentence.txt) for the input format.
* `--output_file`: the path of the output file. Words are segmented by a space; POS labels are attached to the resulting words by an underline ("_").
* `--eval_model`: the pre-trained WMSeg model to be used to segment the sentences in the input file.

To run a pre-trained TwASP model, you need to install SCT and BNP to obtain the auto-analyzed syntactic knowledge. See [data_processing](./data_preprocessing) for more information to download the two toolkits.

## To-do List

* Regular maintenance

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).

