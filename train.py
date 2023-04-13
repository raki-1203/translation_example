import argparse
import datetime
import os
import time
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from transformers import set_seed, BertTokenizerFast

from utils.model.vanila_transformer import VanilaTransformer
from utils.trainer import VanilaTransformerTrainer


def parse_args():
    # Parsing input arguments
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a translation task')
    parser.add_argument(
        '--src_lang',
        type=str,
        default=None,
        help='Source language id for translation.',
    )
    parser.add_argument(
        '--tgt_lang',
        type=str,
        default=None,
        help='Target language id for translation.',
    )
    parser.add_argument(
        '--sample_ratio',
        type=float,
        default=None,
        help='data sample ratio',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=30,
        help='number of epochs',
    )
    parser.add_argument(
        '--num_encoder_layers',
        type=int,
        default=6,
        help='number of transformer encoder layer',
    )
    parser.add_argument(
        '--num_decoder_layers',
        type=int,
        default=6,
        help='number of transformer decoder layer',
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='transformer hidden dimension',
    )
    parser.add_argument(
        '--nhead',
        type=int,
        default=8,
        help='number of transformer attention head',
    )
    parser.add_argument(
        '--dim_feedforward',
        type=int,
        default=2048,
        help='transformer feed forward network hidden dimension',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help=(
            'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, '
            'sequences shorter will be padded if `--pad_to_max_length` is passes.'
        ),
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.06,
        help='warmup proportion',
    )
    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='Adam Optimizer beta 1',
    )
    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.98,
        help='Adam Optimizer beta 2',
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-9,
        help='Adam Optimizer epsilon',
    )
    parser.add_argument('--log_dir', type=str, default='./log/transformer', help='log directory')
    parser.add_argument('--save_dir', type=str, default='./saved_model/transformer', help='saved model directory')
    parser.add_argument('--tokenizer_dir', type=str, default='./tokenizer', help='saved tokenizer directory')
    parser.add_argument('--seed', type=int, default=42, help='A seed for reproducible training.')
    parser.add_argument('--run_name', type=str, default='', help='wandb run name')
    parser.add_argument(
        '--report_to',
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`, '
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            'Only applicable when `--with_tracking` is passed.'
        ),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    # load dataset
    src_tgt_dataset = load_dataset(f'Heerak/{args.src_lang}_{args.tgt_lang}_translation',
                                   cache_dir='/data/heerak/.cache')

    # load tokenizer
    src_vocab_path = os.path.join(args.tokenizer_dir, args.src_lang)
    src_tokenizer = BertTokenizerFast.from_pretrained(src_vocab_path,
                                                      unk_token='[UNK]',
                                                      sep_token='[EOS]',
                                                      pad_token='[PAD]',
                                                      cls_token='[SOS]',
                                                      mask_token='[MASK]',
                                                      model_max_length=args.max_length,
                                                      strip_accents=False,
                                                      lowercase=False)
    tgt_vocab_path = os.path.join(args.tokenizer_dir, args.tgt_lang)
    tgt_tokenizer = BertTokenizerFast.from_pretrained(tgt_vocab_path,
                                                      unk_token='[UNK]',
                                                      sep_token='[EOS]',
                                                      pad_token='[PAD]',
                                                      cls_token='[SOS]',
                                                      mask_token='[MASK]',
                                                      model_max_length=args.max_length,
                                                      strip_accents=False,
                                                      lowercase=False)

    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load model
    model = VanilaTransformer(num_encoder_layers=args.num_encoder_layers,
                              num_decoder_layers=args.num_decoder_layers,
                              d_model=args.d_model,
                              nhead=args.nhead,
                              src_vocab_size=src_vocab_size,
                              tgt_vocab_size=tgt_vocab_size,
                              dim_feedforward=args.dim_feedforward)
    model.to(device)

    # xavier_uniform 파라미터 초기화
    for n, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # load loss_fn & optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           betas=(args.adam_beta1, args.adam_beta2), eps=args.eps)

    if args.report_to == 'wandb':
        name = f'{args.run_name} - {datetime.datetime.now().strftime("%Y%m%d %H%M%S")}'
        wandb.init(project=f'{args.src_lang}-{args.tgt_lang} translation',
                   name=name,
                   config=vars(args))

    # load trainer
    trainer = VanilaTransformerTrainer(args=args,
                                       model=model,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       device=device,
                                       src_tokenizer=src_tokenizer,
                                       tgt_tokenizer=tgt_tokenizer,
                                       dataset=src_tgt_dataset)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Start Training
    train_losses = []
    valid_losses = []

    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss = trainer.train_epoch()
        end_time = time.time()

        valid_loss = trainer.evaluate()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}\n'
              f'Epoch time = {(end_time - start_time):.3f}s')

        # 에폭마다 loss 의 history 를 남김
        np.savetxt(os.path.join(args.log_dir, 'loss_history.txt'), np.array([train_losses, valid_losses]), fmt='%.4e')

        if not os.path.exists(os.path.join(args.save_dir, args.run_name)):
            os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)

        if epoch == 1:
            LEAST_VALID_LOSS = valid_loss
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(args.save_dir, args.run_name, 'best_model.pt'))
        else:
            if LEAST_VALID_LOSS > valid_loss:
                print('모델 갱신 : valid loss {}'.format(valid_loss))
                LEAST_VALID_LOSS = valid_loss
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(args.save_dir, args.run_name, 'best_model.pt'))

        if args.report_to == 'wandb':
            wandb.log({
                'train/loss': train_loss,
                'train/learning_rate': trainer.scheduler.optimizer.param_groups[0]['lr'],
                'eval/loss': valid_loss,
                'eval/best_loss': LEAST_VALID_LOSS,
            })


if __name__ == '__main__':
    main()
