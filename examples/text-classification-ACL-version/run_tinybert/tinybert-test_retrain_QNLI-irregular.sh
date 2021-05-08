export GLUE_DIR=data/glue
export TASK_NAME=QNLI
export PENALTY=penalty_tinybert-4
export PRUNE_RATIO=tinybert_prune_ratios_30
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path /data/ZLKong/tinybert/prune-irregular/QNLI \
  --task_name $TASK_NAME \
  --do_train \
  --masked_retrain \
  --lr_retrain 2e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 174 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --num_train_epochs 5.0 \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 8 \
  --sparsity_type irregular \
  --output_dir /data/ZLKong/tinybert/retrain-irregular/QNLI_30 \
  --overwrite_output_dir \
  --logging_steps 2000 \
  --logging_dir /data/ZLKong/tinybert/retrain-irregular/QNLI_30 \
  --save_steps 40000
