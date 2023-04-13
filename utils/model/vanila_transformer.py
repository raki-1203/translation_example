import math

import torch
import torch.nn as nn

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # 일종의 layer 로서 작용하지만, optimizer 에 의해 업데이트 되지 않도록 함
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class VanilaTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super(VanilaTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # 토큰 임베딩을 거친 후 positional encoding 을 수행하여 임베딩 벡터를 만듦
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Seq2Seq 트랜스포머 네트워크를 통과함
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

        # 디코더 결과를 linear layer 에 통과시켜 target vocab size 만큼 차원을 맞추어 줌
        return self.linear(outs)

    def encode(self,
               src: Tensor,
               src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self,
               tgt: Tensor,
               memory: Tensor,
               tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

    def greedy_decode(self, src, max_len):
        # Set model to evaluation
        self.model.eval()

        # Run Encoder on complete input sequence
        src_mask = torch.zeros((src.shape[-1], src.shape[-1]), device=self.device).to(torch.bool)
        memory = self.encode(src, src_mask)

        # Shape: (batch, sequence_length, hiden_dim
        tgt = ['[SOS]'] * src.shape[0]
        tgt = torch.tensor(self.tgt_tokenizer.batch_encode_plus(tgt, add_special_tokens=False)['input_ids']).to(self.device)
        print('Initial tgt shape:', tgt.shape)

        # Iterate max sequence length
        for i in range(max_len - 1):
            # Avoid the decoder self_attention to attend to future
            tgt_mask = (torch.triu(torch.ones((tgt.shape[-1], tgt.shape[-1]), device=self.device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

            decoder_output = self.model.decode(tgt, memory, tgt_mask)
            output = self.model.linear(decoder_output[:, -1:, :])

            _, next_token = torch.max(output, dim=-1)

            # concat tgt, next_token
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token == 3:  # '[EOS]' 토큰 나온 경우 멈춤
                break

        return tgt

