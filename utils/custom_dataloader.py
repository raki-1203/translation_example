import torch

from torch.utils.data import DataLoader


def get_dataloader(args, dataset, src_tokenizer, tgt_tokenizer):
    if args.sample_ratio is not None:
        train_sample = int(len(dataset['train']) * args.sample_ratio)
        valid_sample = int(len(dataset['valid']) * args.sample_ratio)
    else:
        train_sample = len(dataset['train'])
        valid_sample = len(dataset['valid'])
    train_dataset = dataset['train'].shuffle(args.seed).select(range(train_sample))
    valid_dataset = dataset['valid'].shuffle(args.seed).select(range(valid_sample))

    data_collator = TransformerDataCollator()

    def preprocess_function(examples):
        inputs = [ex for ex in examples[args.src_lang]]
        targets = [ex for ex in examples[args.tgt_lang]]
        tokenized_inputs = src_tokenizer(inputs, max_length=256, padding='max_length', truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        tokenized_targets = tgt_tokenizer(text_target=targets, max_length=256, padding='max_length', truncation=True)

        result = {}
        result['src_input_ids'] = tokenized_inputs['input_ids']
        result['tgt_input_ids'] = tokenized_targets['input_ids']

        # if we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        result['labels'] = [[(t if t != tgt_tokenizer.pad_token_id else -100) for t in tokenized_target]
                            for tokenized_target in tokenized_targets['input_ids']]

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

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  shuffle=True,
                                  batch_size=args.batch_size)

    valid_dataloader = DataLoader(valid_dataset,
                                  collate_fn=data_collator,
                                  shuffle=False,
                                  batch_size=args.batch_size)

    return {'train': train_dataloader,
            'valid': valid_dataloader}


class TransformerDataCollator:

    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, batch):
        b_src = []
        if not self.is_test:
            b_tgt = []
            b_labels = []

        for b in batch:
            b_src.append(b['src_input_ids'])

            if not self.is_test:
                b_tgt.append(b['tgt_input_ids'])
                b_labels.append(b['labels'])

        t_src = torch.LongTensor(b_src)  # List[Tensor] -> Tensor List
        if self.is_test:
            return {'src': t_src}
        else:
            t_tgt = torch.LongTensor(b_tgt)
            t_labels = torch.LongTensor(b_labels)
            return {'src': t_src, 'tgt': t_tgt, 'labels': t_labels}
