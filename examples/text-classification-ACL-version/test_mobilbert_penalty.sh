export GLUE_DIR=data/glue
export TASK_NAME=RTE


CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-1 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-2 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty2 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty2 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-3 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty3 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty3 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-4 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty4 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty4 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-5 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty5 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty5 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=2 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-6 \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 384 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --evaluate_during_training \
  --sparsity_type column \
  --block_row_division 2 \
  --block_row_width 8 \
  --output_dir /data/ZLKong/mobilebert/test_penalty6 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty6 \
  --save_steps 4000  



