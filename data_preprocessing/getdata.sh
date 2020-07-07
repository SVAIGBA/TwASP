############## process data ##############

# download Universal Dependencies 2.4
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz

tar zxvf ud-treebanks-v2.4.tgz
rm ud-treebanks-v2.4.tgz

## process UD
python data_preprocessing.py --dataset=ud --translate

## process CTB5
python data_preprocessing.py --dataset=ctb5

## process CTB6
python data_preprocessing.py --dataset=ctb6

## process CTB7
python data_preprocessing.py --dataset=ctb7

## process CTB9
python data_preprocessing.py --dataset=ctb9

############### process data ##############


############### download Stanford CoreNLP #################
# download StanfordCoreNLP v3.9.2
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip

unzip stanford-corenlp-full-2018-10-05.zip
rm stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05 || exit
# download Chinese
wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-10-05-models.jar

cd ..

############## download Stanford CoreNLP #################


############## download Berkeley Neural Parser (BNP) #################
# You can see https://github.com/nikitakit/self-attentive-parser for details

pip install cython numpy
pip install benepar[cpu]
pip install nltk

# download BNP model
python install_BNP.py

############## download Berkeley Neural Parser #################

cd ..

############## obtain auto_processed data from Stanford CoreNLP Toolkits (SCT) #################

python get_syninfo.py --dataset=UD1 --toolkit=SCT --overwrite

python get_syninfo.py --dataset=UD2 --toolkit=SCT --overwrite

python get_syninfo.py --dataset=CTB5 --toolkit=SCT --overwrite

python get_syninfo.py --dataset=CTB6 --toolkit=SCT --overwrite

python get_syninfo.py --dataset=CTB7 --toolkit=SCT --overwrite

python get_syninfo.py --dataset=CTB8 --toolkit=SCT --overwrite

############## obtain auto_processed data from Stanford CoreNLP Toolkits (SCT) #################



############## obtain auto_processed data from Berkeley Neural Parser (BNP) #################

# BNP requires TensorFlow framework (tensorflow-gpu=1.13). It is recommended to use GPUs when obtaining syntactic knowledge from BNP

# uncomment the following line if you want to install Tensorflow to run BNP
# pip install tensorflow-gpu==1.13.1

python get_syninfo.py --dataset=UD1 --toolkit=BNP --overwrite

python get_syninfo.py --dataset=UD2 --toolkit=BNP --overwrite

python get_syninfo.py --dataset=CTB5 --toolkit=BNP --overwrite

python get_syninfo.py --dataset=CTB6 --toolkit=BNP --overwrite

python get_syninfo.py --dataset=CTB7 --toolkit=BNP --overwrite

python get_syninfo.py --dataset=CTB8 --toolkit=BNP --overwrite

############## obtain auto_processed data from Berkeley Neural Parser (BNP) #################
