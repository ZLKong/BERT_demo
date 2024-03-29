export GLUE_DIR=data/glue
export TASK_NAME=MNLI
export OUT_DIR=/data/ZLKong/mobilebert-py3.6-2/finetune/MNLI

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
          --model_name_or_path google/mobilebert-uncased \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 3e-5 \
          --num_train_epochs 4.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 20000 \
          --logging_dir $OUT_DIR \
          --save_steps 50000
