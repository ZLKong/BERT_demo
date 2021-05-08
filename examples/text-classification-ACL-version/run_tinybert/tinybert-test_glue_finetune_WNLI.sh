export GLUE_DIR=data/glue
export TASK_NAME=WNLI
#export INT_DIR=mrm8488/es-tinybert-v1-1
export INT_DIR=/data/ZLKong/General_TinyBERT_4L_312D
#export INT_DIR=/data/ZLKong/TinyBERT_4L_312D/MRPC
export OUT_DIR=/data/ZLKong/tinybert/finetune/WNLI
#export OUT_DIR=/data/ZLKong/tinybert/finetune/MRPC-testtrainerwithprune

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
          --model_name_or_path $INT_DIR \
          --task_name $TASK_NAME \
	  --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 3e-5 \
          --num_train_epochs 5.0 \
          --output_dir $OUT_DIR \
          --overwrite_output_dir \
          --logging_steps 10000 \
          --logging_dir $OUT_DIR \
          --save_steps 40000
