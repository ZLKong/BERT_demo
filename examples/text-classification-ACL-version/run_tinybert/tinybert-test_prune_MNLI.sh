export GLUE_DIR=data/glue
export TASK_NAME=MNLI
export PENALTY=penalty_tinybert-4-2
export INT_DIR=/data/ZLKong/TinyBERT_4L_312D/MNLI
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path $INT_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file ${PENALTY} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 4e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/tinybert/prune/MNLI \
  --overwrite_output_dir \
  --logging_steps 2000 \
  --logging_dir /data/ZLKong/tinybert/prune/MNLI \
  --save_steps 40000
