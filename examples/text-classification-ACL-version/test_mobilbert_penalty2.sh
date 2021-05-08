export GLUE_DIR=data/glue
export TASK_NAME=RTE


CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-7 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty7 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty7 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-8 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty8 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty8 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-9 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty9 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty9 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-10 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty10 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty10 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-11 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty11 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty11 \
  --save_steps 4000  &&

CUDA_VISIBLE_DEVICES=3 python run_glue.py \
  --model_name_or_path /data/ZLKong/mobilebert/finetune/RTE689 \
  --task_name $TASK_NAME \
  --do_train \
  --rew \
  --penalty_config_file penalty_test/glue-penalty_mobilebert-12 \
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
  --output_dir /data/ZLKong/mobilebert/test_penalty12 \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --logging_dir /data/ZLKong/mobilebert/test_penalty12 \
  --save_steps 4000  



