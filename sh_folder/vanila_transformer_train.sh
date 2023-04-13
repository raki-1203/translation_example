# loss 가 계속 오르는 문제.......
#python train.py \
#--src_lang en \
#--tgt_lang ko \
#--num_epochs 30 \
#--num_encoder_layers 6 \
#--num_decoder_layers 6 \
#--d_model 512 \
#--nhead 8 \
#--dim_feedforward 2048 \
#--batch_size 16 \
#--max_length 256 \
#--learning_rate 1e-4 \
#--adam_beta1 0.9 \
#--adam_beta2 0.98 \
#--eps 1e-9 \
#--log_dir ./log/transformer \
#--save_dir ./saved_model/transformer \
#--tokenizer_dir ./tokenizer \
#--seed 42 \
#--report_to wandb \
#--run_name 'Vanila Transformer'

# train datsaet 30%, valid dataset 30%, learning_rate 더 작게
python train.py \
--src_lang en \
--tgt_lang ko \
--sample_ratio .3 \
--num_epochs 30 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--d_model 512 \
--nhead 8 \
--dim_feedforward 2048 \
--batch_size 16 \
--max_length 256 \
--learning_rate 5e-5 \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--eps 1e-9 \
--log_dir ./log/transformer \
--save_dir ./saved_model/transformer \
--tokenizer_dir ./tokenizer \
--seed 42 \
--report_to wandb \
--run_name 'Vanila Transformer Sample 30%'