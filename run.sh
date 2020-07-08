mkdir logs

# important parameters
# do_train: train the model
# do_test: test the model
# use_bert: use BERT as encoder
# use_zen: use ZEN as encoder
# bert_model: the directory of BERT/ZEN model
# use_attention: use two-way attention
# source: the toolkit to be use (stanford or berkeley)
# feature_flag: use pos, chunk, or dep knowledge
# model_name: the name of model to save

# training

# Command lines to train our model with POS knowledge from SCT
# use BERT

# CTB5
python twasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --eval_data_path=./data/CTB5/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB5_bert_two_way_stanford_pos

# CTB6
python twasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --eval_data_path=./data/CTB6/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB6_bert_two_way_stanford_pos

# CTB7
python twasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --eval_data_path=./data/CTB7/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB7_bert_two_way_stanford_pos

# CTB9
python twasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --eval_data_path=./data/CTB9/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB9_bert_two_way_stanford_pos

# UD1
python twasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_bert_two_way_stanford_pos

# UD2
python twasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_bert_two_way_stanford_pos


# use ZEN

# CTB5
python twasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --eval_data_path=./data/CTB5/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB5_zen_two_way_stanford_pos

# CTB6
python twasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --eval_data_path=./data/CTB6/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB6_zen_two_way_stanford_pos

# CTB7
python twasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --eval_data_path=./data/CTB7/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB7_zen_two_way_stanford_pos

# CTB9
python twasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --eval_data_path=./data/CTB9/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=CTB9_zen_two_way_stanford_pos

# UD1
python twasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --eval_data_path=./data/UD1/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD1_zen_two_way_stanford_pos

# UD2
python twasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --eval_data_path=./data/UD2/dev.tsv --use_zen --bert_model=/path/to/zen/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=100 --warmup_proportion=0.1 --learning_rate=1e-5  --patient=15 --source=stanford --feature_flag=pos --model_name=UD2_zen_two_way_stanford_pos


# testing

python twasp_main.py --do_test --eval_data_path=./data/dataset_name/test.tsv --eval_model=./models/model_name/model.pt

