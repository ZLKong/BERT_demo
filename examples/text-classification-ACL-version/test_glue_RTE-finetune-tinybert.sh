export GLUE_DIR=data/glue
export TASK_NAME=RTE
export OUT_DIR=/tmp/fintune_RTE_mobilebert/

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
          --model_name_or_path 1757968399/tinybert_4_312_1200 \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 384 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 5e-3 \
          --num_train_epochs 3.0 \
	  --max_grad_norm=1.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 400 \
          --logging_dir $OUT_DIR \
          --save_steps 400
