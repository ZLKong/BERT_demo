export GLUE_DIR=data/glue
export TASK_NAME=WNLI
export OUT_DIR=/data/ZLKong/irregular/results/roberta-finetune/WNLI

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
          --model_name_or_path roberta-base \
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
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 20 \
          --logging_dir $OUT_DIR \
          --save_steps 1000
