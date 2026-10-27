import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from formatter import format_prompt, format_response_header


class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.pad_id = tokenizer.eot_token
        self.samples = []

        for item in data:
            instruction = item["instruction"]
            inp         = item.get("input", "")
            response    = item["output"]

            # Full prompt including response + EOT
            full_text = format_prompt(instruction, inp, response)
            full_ids = tokenizer.encode(full_text) + [self.pad_id]

            # Find where the response starts
            header_text = format_response_header(instruction, inp)
            header_len = len(tokenizer.encode(header_text))

            # Truncate if exceeding max_length
            full_ids = full_ids[:max_length]

            # If truncation eliminated the entire response, skip this sample!
            if len(full_ids) <= header_len:
                continue

            # Build labels: -100 for instruction/prompt, real ids for response
            labels = [-100] * len(full_ids)
            for i in range(header_len, len(full_ids)):
                labels[i] = full_ids[i]

            self.samples.append((full_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100):
    """
    Dynamically pads the batch to the maximum length of samples IN THIS BATCH ONLY.
    Saves massive GPU memory and speed by avoiding unnecessary padding.
    """
    batch_max_len = max(len(ids) for ids, _ in batch)

    batch_input_ids = []
    batch_labels = []

    for ids, labels in batch:
        pad_len = batch_max_len - len(ids)
        padded_ids = ids + [pad_token_id] * pad_len
        padded_labels = labels + [ignore_index] * pad_len

        batch_input_ids.append(torch.tensor(padded_ids, dtype=torch.long))
        batch_labels.append(torch.tensor(padded_labels, dtype=torch.long))

    return torch.stack(batch_input_ids), torch.stack(batch_labels)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataloaders(
    train_path="data/train.json",
    val_path="data/val.json",
    batch_size=4,
    max_length=512,
):
    import os
    from create_splits import create_reproducible_splits

    # Auto-generate splits if they do not exist
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Split files not found. Creating reproducible splits from alpaca_data.json ...")
        create_reproducible_splits()

    tokenizer = tiktoken.get_encoding("gpt2")
    train_data = load_json(train_path)
    val_data = load_json(val_path)

    train_ds = AlpacaDataset(train_data, tokenizer, max_length)
    val_ds   = AlpacaDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: custom_collate_fn(b, pad_token_id=tokenizer.eot_token)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: custom_collate_fn(b, pad_token_id=tokenizer.eot_token)
    )

    print(f"Loaded explicit splits -> Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, tokenizer


    


