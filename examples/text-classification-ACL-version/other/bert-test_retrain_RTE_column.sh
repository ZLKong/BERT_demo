export GLUE_DIR=data/glue
export TASK_NAME=RTE
export PENALTY=penalty_mobilebert-4
export PRUNE_RATIO=bert_prune_ratios_50
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=6 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/prune/RTE_bert_column \
  --task_name $TASK_NAME \
  --do_train \
  --masked_retrain \
  --lr_retrain 5e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --num_train_epochs 1 \
  --sparsity_type column \
  --evaluate_during_training \
  --block_row_division 3 \
  --block_row_width 3 \
  --output_dir /data/ZLKong/mobilebert/retrain/RTE_bert_column \
  --overwrite_output_dir \
  --logging_steps 1500 \
  --logging_dir /data/ZLKong/mobilebert/retrain/RTE_bert_column \
  --save_steps 4000
