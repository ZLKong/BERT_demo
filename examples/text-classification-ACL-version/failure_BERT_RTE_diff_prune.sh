export GLUE_DIR=data/glue
export TASK_NAME=MNLI
#export INT_DIR=/data/ZLKong/irregular/results/bert-finetune/MNLI
#export INT_DIR=/data/ZLKong/irregular/results/bert-retrain/RTE_55
#export INT_DIR=/data/ZLKong/irregular/results/bert-retrain/RTE_70
export INT_DIR=/data/ZLKong/irregular/results/bert-retrain/MNLI_50
export OUT_DIR=/data/ZLKong/bert/finetune/failure

CUDA_VISIBLE_DEVICES=7 python geng_dac_differential.py \
          --model_name_or_path $INT_DIR \
          --task_name $TASK_NAME \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 5e-5 \
          --num_train_epochs 1.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 150 \
          --logging_dir $OUT_DIR \
          --save_steps 4000                    
