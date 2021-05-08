export TASK_NAME=MRPC


CUDA_VISIBLE_DEVICES=6 python run_glue.py \
          --model_name_or_path  huawei-noah/TinyBERT_General_4L_312D \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 3.0 \
          --output_dir results/$TASK_NAME \
          --overwrite_output_dir
