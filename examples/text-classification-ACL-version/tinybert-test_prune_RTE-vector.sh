export GLUE_DIR=data/glue
export TASK_NAME=RTE
export PENALTY=penalty_tinybert-4
#mv output_distilbert_128/$TASK_NAME/pretrain/checkpoint-10/pytorch_model.bin output_distilbert_128/$TASK_NAME/pretrain/
# mv output_distilbert_128/WN$TASK_NAMELI/pretrain/checkpoint-10/config.json output_distilbert_128/$TASK_NAME/pretrain/
CUDA_VISIBLE_DEVICES=1 python run_glue.py \
  --model_name_or_path /data/ZLKong/tinybert/finetune/RTE \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file ${PENALTY} \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --evaluate_during_training \
  --sparsity_type vector \
  --block_row_division 1 \
  --block_row_width 12 \
  --output_dir /data/ZLKong/bert_vector/tinybert-prune/RTE \
  --overwrite_output_dir \
  --logging_steps 20000 \
  --logging_dir  /data/ZLKong/bert_vector/tinybert-prune/RTE \
  --save_steps 40000
