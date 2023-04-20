import torch

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils.custom_dataloader import get_vanila_transformer_dataloader


class VanilaTransformerTrainer:

    def __init__(self, args, model, optimizer, loss_fn, device, src_tokenizer, tgt_tokenizer, dataset):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = args.batch_size
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang
        self.device = device

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        dataloader = get_vanila_transformer_dataloader(args, dataset, src_tokenizer, tgt_tokenizer)

        self.train_dataloader = dataloader['train']
        self.valid_dataloader = dataloader['valid']

        step_per_epoch = len(self.train_dataloader)
        t_total = step_per_epoch * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(optimizer, round(args.warmup_proportion * t_total), t_total)

    def train_epoch(self):
        self.model.train()
        losses = 0

        pbar = tqdm(enumerate(self.train_dataloader, start=1), total=len(self.train_dataloader))
        for idx, batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            labels = batch['labels'].to(self.device)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

            logits = self.model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()

            # Transformer Network 의 teacher forcing 을 아래와 같이 구현함
            # 즉, 학습시 auto regressive 한 디코더의 특징을 활용하는 것이 아니라,
            # t-1 시점의 예측과 무관히 t 시점 디코더에는 ground truth 토큰을 입력함

            # tgt_input 은 [SOS] 토큰으로 '정답 sequence 인 tgt_out 의 첫 토큰'을 맞추려 함
            # tgt_input 의 마지막 토큰으로 '정답 sequence 인 tgt_out 의 마지막 토큰'인 [EOS] 를 맞추려 함
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()

            pbar.set_description(f'train loss: {losses/idx}')

        return losses / len(self.train_dataloader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = 0

        pbar = tqdm(enumerate(self.valid_dataloader, start=1), total=len(self.valid_dataloader))
        for idx, batch in pbar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            labels = batch['labels'].to(self.device)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

            logits = self.model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            # tgt_input 은 [SOS] 토큰으로 '정답 sequence 인 tgt_out 의 첫 토큰'을 맞추려 함
            # tgt_input 의 마지막 토큰으로 '정답 sequence 인 tgt_out 의 마지막 토큰'인 [EOS] 를 맞추려 함
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            losses += loss.item()

            pbar.set_description(f'Valid loss: {losses / idx}')

        return losses / len(self.valid_dataloader)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)

        """
        (eg.) 타겟 시퀀스 길이가 5 토큰이라고 가정하면,
        >>> mask
        tensor([[True, False, False, False, False],  # 첫 토큰은 자기 자신만 attend 할 수 있음
                [True,  True, False, False, False],
                [True,  True,  True, False, False],
                [True,  True,  True,  True, False],
                [True,  True,  True,  True,  True]])  # 마지막 토큰은 모든 토큰을 attend 할 수 있음
        """

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        """
        (eg.) 타겟 시퀀스 길이가 5 토큰이라고 가정하면,
        >>> mask
        tensor([[0, -inf, -inf, -inf, -inf],  # 첫 토큰은 자기 자신만 attend 할 수 있음
                [0,    0, -inf, -inf, -inf],
                [0,    0,    0, -inf, -inf],
                [0,    0,    0,    0, -inf],
                [0,    0,    0,    0,    0]])  # 마지막 토큰은 모든 토큰을 attend 할 수 있음
        """

        return mask

    def create_mask(self, src, tgt):
        """ mask 되지 않고 attend 할 수 있는 토큰을 False 또는 0 값으로 이루어지도록 처리함 """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)
        # src_mask 는 (src 시퀀스길이 X src 시퀀스길이)의 False 들로 이루어진 행렬임)

        src_padding_mask = (src == self.src_tokenizer.pad_token_id)
        tgt_padding_mask = (tgt == self.tgt_tokenizer.pad_token_id)

        """
        (eg.)
        src_padding_mask 및 tgt_padding_mask 는 아래와 같이 batch 로 묶인 데이터들에 대해서 pad_token_id 에 해당하는 부분은 False 로 처리함

        tensor([[False, False, False, False, False, True, True, True, True],
                [False, False, False, False, True, True, True, True, True],
                [False, False, False, False, False, False, False, False, True]])
        """

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
