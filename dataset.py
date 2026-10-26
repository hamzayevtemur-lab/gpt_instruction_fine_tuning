import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from formatter import format_prompt, format_response_header


class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.pad_id=tokenizer.eot_token
        self.samples=[]

        for item in data:
            instruction = item["instruction"]
            inp         = item.get("input", "")
            response    = item["output"]

            # full prompt including response
            full_text=format_prompt(instruction, inp, response)
            full_ids=tokenizer.encode(full_text)+[self.pad_id]

            # Find where the response starts
            header_text=format_response_header(instruction, inp)
            header_len=len(tokenizer.encode(header_text))

            # Truncate
            full_ids=full_ids[:max_length]

            # Build labels: -100 for instruction, real ids for response
            labels=[-100]*len(full_ids)
            for i in range(header_len, len(full_ids)):
                labels[i]=full_ids[i]

            # Pad to max_length
            pad_len=max_length - len(full_ids)
            full_ids=full_ids+ [self.pad_id]* pad_len
            labels=labels+ [-100]* pad_len

            self.samples.append((full_ids, labels))
    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels=self.samples[idx]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )




def load_alpaca(path="data/alpaca_data.json"):
    with open(path, "r") as f:
        return json.load(f)


def get_dataloaders(batch_size=4, max_length=512, val_ratio=0.05):
    tokenizer = tiktoken.get_encoding("gpt2")
    data      = load_alpaca()

    split        = int(len(data) * (1 - val_ratio))
    train_ds     = AlpacaDataset(data[:split], tokenizer, max_length)
    val_ds       = AlpacaDataset(data[split:], tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, tokenizer


    


