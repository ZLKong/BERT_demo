export GLUE_DIR=data/glue
#export GLUE_DIR=
export TASK_NAME=RTE
#export INT_DIR=mrm8488/es-tinybert-v1-1
#export INT_DIR=bert-base-uncased
#export INT_DIR=/data/ZLKong/mobilebert/retrain/RTE_bert_blkfilter
export INT_DIR=bert-base-uncased
export OUT_DIR=/data/ZLKong/bert/finetune/RTE-128-16-5e-5-7epo
#export OUT_DIR=/data/ZLKong/tinybert/finetune/MRPC-testtrainerwithprune

CUDA_VISIBLE_DEVICES=4 python run_glue.py \
          --model_name_or_path $INT_DIR \
          --task_name $TASK_NAME \
	  --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 16 \
          --per_gpu_eval_batch_size 16 \
          --learning_rate 5e-5 \
          --num_train_epochs 3.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 150 \
          --logging_dir $OUT_DIR \
          --save_steps 4000
