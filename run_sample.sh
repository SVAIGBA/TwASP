mkdir logs

# train
python joint_main.py --do_train --train_data_path=./sample_data/train.tsv --eval_data_path=./sample_data/test.tsv --use_bert --bert_model=/path/to/bert/model --use_attention --max_seq_length=300 --max_ngram_size=300  --train_batch_size=2 --eval_batch_size=2 --num_train_epochs=3 --warmup_proportion=0.1 --learning_rate=1e-5 --patient=15 --source=stanford --feature_flag=pos --model_name=sample_model

# test
python joint_main.py --do_test --eval_data_path=./sample_data/test.tsv --eval_model=./models/model_name/model.pt

