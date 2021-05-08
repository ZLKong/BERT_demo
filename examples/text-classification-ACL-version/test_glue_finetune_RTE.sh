export GLUE_DIR=data/glue
export TASK_NAME=RTE
export OUT_DIR=/data/ZLKong/irregular/results/L-12_H-256_A-4/RTE

CUDA_VISIBLE_DEVICES=1 python run_glue.py \
          --model_name_or_path google/bert_uncased_L-12_H-256_A-4 \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 64 \
          --per_gpu_eval_batch_size 64 \
          --learning_rate 1e-4 \
          --num_train_epochs 4.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 300 \
          --logging_dir $OUT_DIR \
          --save_steps 4000
