# google/mt5-small
python run_translation.py \
--output_dir ./saved_model/mt5_small_en_ko_all \
--do_train True \
--do_eval True \
--do_predict True \
--evaluation_strategy steps \
--num_train_epochs 5 \
--seed 42 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 32 \
--load_best_model_at_end True \
--metric_for_best_model eval_bleu \
--greater_is_better True \
--sortish_sampler True \
--predict_with_generate True \
--model_name_or_path google/mt5-small \
--cache_dir /data/heerak/.cache \
--source_lang en \
--target_lang ko \
--dataset_name Heerak/ko_en_parallel_dataset \
--num_beams 5 \
--report_to wandb \
--run_name mt5_small_en_ko_all \
--log_level warning \
--eval_steps 10000 \
--evaluation_strategy steps \
--save_steps 10000 \
--save_total_limit 2 \
--preprocessing_num_workers 4 \
#--max_train_samples 10000 \
#--max_eval_samples 1000 \


# google/mt5-small
#python run_translation_no_trainer.py \
#--dataset_name Heerak/en_ko_translation \
#--cache_dir /data/heerak/.cache \
#--predict_with_generate True \
#--num_beams 5 \
#--max_source_length 512 \
#--max_target_length 512 \
#--source_lang en \
#--target_lang ko \
#--source_prefix "translate English to Korean: " \
#--preprocessing_num_workers 4 \
#--overwrite_cache \
#--model_name_or_path google/mt5-small \
#--use_slow_tokenizer \
#--per_device_train_batch_size 8 \
#--per_device_eval_batch_size 16 \
#--learning_rate 5e-5 \
#--num_train_epochs 5 \
#--gradient_accumulation_steps 2 \
#--num_warmup_steps 1000 \
#--output_dir ./saved_model/mt5_small_en_ko_test \
#--seed 42 \
#--checkpointing_steps 500 \
#--with_tracking \
#--report_to wandb \
#--max_train_samples 10000 \
#--max_eval_samples 10000