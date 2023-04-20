import os
import argparse

import torch

from transformers import BertTokenizerFast, set_seed
from datasets import load_dataset

from utils.model.vanila_transformer import VanilaTransformer
from utils.custom_dataloader import get_vanila_transformer_dataloader


class Arguments:
    src_lang = 'en'
    tgt_lang = 'ko'
    sample_ratio = 0.0001
    num_epochs = 30
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    batch_size = 16
    max_length = 256
    learning_rate = 1e-4
    warmup_proportion = 0.06
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    eps = 1e-9
    log_dir = './log/transformer'
    save_dir = './saved_model/transformer'
    tokenizer_dir = './tokenizer'
    seed = 42
    run_name = ''
    report_to = None


def greedy_decode(model, src, max_len, device, tgt_tokenizer):
    # Set model to evaluation
    model.eval()

    # Run Encoder on complete input sequence
    src_mask = torch.zeros((src.shape[-1], src.shape[-1]), device=device).to(torch.bool)
    memory = model.encode(src, src_mask)

    # Shape: (batch, sequence_length, hiden_dim
    tgt = ['[SOS]'] * src.shape[0]
    tgt = torch.tensor(tgt_tokenizer.batch_encode_plus(tgt, add_special_tokens=False)['input_ids']).to(device)
    print('Initial tgt shape:', tgt.shape)

    # Iterate max sequence length
    for i in range(max_len - 1):
        # Avoid the decoder self_attention to attend to future
        tgt_mask = (torch.triu(torch.ones((tgt.shape[-1], tgt.shape[-1]), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        decoder_output = model.decode(tgt, memory, tgt_mask)
        output = model.linear(decoder_output[:, -1:, :])

        _, next_token = torch.max(output, dim=-1)

        # concat tgt, next_token
        tgt = torch.cat([tgt, next_token], dim=1)

        # if next_token == 3:  # '[EOS]' 토큰 나온 경우 멈춤
        #     break

    return tgt


def main():
    args = Arguments()

    set_seed(args.seed)

    # load dataset
    # src_tgt_dataset = load_dataset(f'Heerak/{args.src_lang}_{args.tgt_lang}_translation',
    #                                cache_dir='/data/heerak/.cache')

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

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    model = VanilaTransformer(num_encoder_layers=6,
                              num_decoder_layers=6,
                              d_model=512,
                              nhead=8,
                              src_vocab_size=src_vocab_size,
                              tgt_vocab_size=tgt_vocab_size,
                              dim_feedforward=2048)
    model.to(device)

    inputs = ['I exercise every day, work, walk the dog, and have so many things to do.',
              'I love you']
    src = torch.tensor(src_tokenizer(inputs, padding=True)['input_ids']).to(device)
    tgt = greedy_decode(model, src, max_len=256, device=device, tgt_tokenizer=tgt_tokenizer)
    print(tgt_tokenizer.batch_decode(tgt, skip_special_tokens=True))


if __name__ == '__main__':
    main()
