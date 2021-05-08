export GLUE_DIR=data/glue
export TASK_NAME=QQP
export PENALTY=penalty_tinybert-4
export PRUNE_RATIO=tinybert_prune_ratios_50
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/tinybert/prune-irregular/QQP \
  --task_name $TASK_NAME \
  --do_train \
  --masked_retrain \
  --lr_retrain 4e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --num_train_epochs 6.0 \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 8 \
  --sparsity_type irregular \
  --output_dir /data/ZLKong/tinybert/retrain-irregular/QQP_50 \
  --overwrite_output_dir \
  --logging_steps 3000 \
  --logging_dir /data/ZLKong/tinybert/retrain-irregular/QQP_50 \
  --save_steps 40000
