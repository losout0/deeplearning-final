import torch, tiktoken
from torch.utils.data import Dataset, DataLoader


class Dataset_GPT(Dataset):
    def __init__(self, text, tokenizer, stride, max_length):
        self.input_ids = []
        self.target_ids = []

        allowed_special = {'<|endoftext|>'}
        tokens = tokenizer.encode(text, allowed_special=allowed_special)

        for i in range(0, len(tokens) - max_length, stride):
            self.input_ids.append(torch.tensor(tokens[i: i + max_length]))
            self.target_ids.append(torch.tensor(tokens[i+1: i+max_length + 1]))

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __len__(self):
        return len(self.input_ids)
    

def create_dataset(text, stride, max_length, shuffle, drop_last, tokenizer, num_workers, batch_size):

    dataset = Dataset_GPT(
        text=text,
        tokenizer=tokenizer,
        stride=stride,
        max_length=max_length
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


def get_loaders(train_data, test_data, val_data, tokenizer, max_length = 256, batch_sz = 10):
    train_loader = create_dataset(
        text=train_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=4,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True
    )

    test_loader = create_dataset(
        text=test_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=4,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True
    )

    val_loader = create_dataset(
        text=val_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=4,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True
    )
    
    return train_loader, test_loader, val_loader