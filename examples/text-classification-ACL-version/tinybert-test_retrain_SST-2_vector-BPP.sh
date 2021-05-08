export GLUE_DIR=data/glue
export TASK_NAME=SST-2
export PENALTY=penalty_tinybert-4
export PRUNE_RATIO=tinybert_prune_ratios_10
#export IN_DIR=/data/ZLKong/mobilebert/retrain/RTE_bert_blkfilter2
export IN_DIR=/data/ZLKong/SST-2/block_size_12_1/model/prune_ratio_10
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=5 python run_glue.py \
  --model_name_or_path $IN_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --masked_retrain \
  --lr_retrain 3e-5 \
  --penalty_config_file ${PENALTY} \
  --prune_ratio_config ${PRUNE_RATIO} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --num_train_epochs 5 \
  --sparsity_type vector \
  --evaluate_during_training \
  --block_row_division 1 \
  --block_row_width 12 \
  --output_dir /data/ZLKong/bert_vector/tinybert-retrain/SST-2-BBP-10 \
  --overwrite_output_dir \
  --logging_steps 20000 \
  --logging_dir /data/ZLKong/bert_vector/tinybert-retrain/SST-2-BBP-10 \
  --save_steps 40000
