export GLUE_DIR=data/glue
export TASK_NAME=RTE
export PENALTY=penalty_mobilebert-4
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file ${PENALTY} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 7.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 8 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/prune/RTE_column \
  --overwrite_output_dir \
  --logging_steps 2000 \
  --logging_dir /data/ZLKong/mobilebert/prune/RTE_column \
  --save_steps 4000
