import torch

from torch.utils.data import DataLoader


def get_vanila_transformer_dataloader(args, dataset, src_tokenizer, tgt_tokenizer):
    if args.sample_ratio is not None:
        train_sample = int(len(dataset['train']) * args.sample_ratio)
        valid_sample = int(len(dataset['valid']) * args.sample_ratio)
    else:
        train_sample = len(dataset['train'])
        valid_sample = len(dataset['valid'])
    train_dataset = dataset['train'].shuffle(args.seed).select(range(train_sample))
    valid_dataset = dataset['valid'].shuffle(args.seed).select(range(valid_sample))
    test_dataset = dataset['test']

    data_collator = VanilaTransformerDataCollator(src_tokenizer, tgt_tokenizer)

    def preprocess_function(examples):
        inputs = [ex for ex in examples[args.src_lang]]
        targets = [ex for ex in examples[args.tgt_lang]]
        tokenized_inputs = src_tokenizer(inputs)

        # Tokenize targets with the `text_target` keyword argument
        tokenized_targets = tgt_tokenizer(text_target=targets)

        result = {'src_input_ids': tokenized_inputs['input_ids'],
                  'tgt_input_ids': tokenized_targets['input_ids']}

        return result

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    valid_dataset = valid_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=valid_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  shuffle=True,
                                  batch_size=args.batch_size)

    valid_dataloader = DataLoader(valid_dataset,
                                  collate_fn=data_collator,
                                  shuffle=False,
                                  batch_size=args.batch_size)

    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=data_collator,
                                 shuffle=False,
                                 batch_size=args.batch_size)

    return {'train': train_dataloader,
            'valid': valid_dataloader,
            'test': test_dataloader}


class VanilaTransformerDataCollator:

    def __init__(self, src_tokenizer, tgt_tokenizer, is_test=False):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.is_test = is_test

    def __call__(self, batch):
        b_src = []
        if not self.is_test:
            b_tgt = []
            b_labels = []

        src_max_sequence = max([len(b['src_input_ids']) for b in batch])
        tgt_max_sequence = max([len(b['tgt_input_ids']) for b in batch])

        for b in batch:
            src_input_ids = b['src_input_ids'] + \
                            [self.src_tokenizer.pad_token_id] * (src_max_sequence - len(b['src_input_ids']))
            b_src.append(src_input_ids)

            if not self.is_test:
                # batch 에 있는 단어에서 마지막 [EOS] 토큰 제거
                tgt_input_ids = b['tgt_input_ids'][:-1] + \
                                [self.tgt_tokenizer.pad_token_id] * (tgt_max_sequence - len(b['tgt_input_ids'][:-1]))
                b_tgt.append(tgt_input_ids)
                # batch 에 있는 단어에서 마지막 [SOS] 토큰 제거
                labels = b['tgt_input_ids'][1:] + [-100] * (tgt_max_sequence - len(b['tgt_input_ids'][1:]))
                b_labels.append(labels)

        t_src = torch.LongTensor(b_src)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'src': t_src}
        else:
            t_tgt = torch.LongTensor(b_tgt)
            t_labels = torch.LongTensor(b_labels)
            return {'src': t_src, 'tgt': t_tgt, 'labels': t_labels}
