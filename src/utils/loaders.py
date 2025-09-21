import os
import torch, tiktoken
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Dataset_GPT(Dataset):
    def __init__(self, text, tokenizer, stride, max_length, set):
        self.input_ids = []
        self.target_ids = []

        allowed_special = {'<|endoftext|>'}
        tokens = tokenizer.encode(text, allowed_special=allowed_special)

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        
        input_ids = self.tokens[start_idx : start_idx + self.max_length]
        target_ids = self.tokens[start_idx + 1 : start_idx + self.max_length + 1]
        
        return input_ids, target_ids

    def __len__(self):
        return (len(self.tokens) - self.max_length - 1) // self.stride
    

def create_dataset(text, stride, max_length, shuffle, drop_last, tokenizer, num_workers, batch_size, set):

    dataset = Dataset_GPT(
        text=text,
        tokenizer=tokenizer,
        stride=stride,
        max_length=max_length,
        set=set
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

def load_file(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        content = file.read()
    return content


def get_loaders(data_path, tokenizer, max_length = 256, batch_sz = 10):
    train_data = load_file(os.path.join(data_path, "train.txt"))
    test_data = load_file(os.path.join(data_path, "test.txt"))
    val_data = load_file(os.path.join(data_path, "val.txt"))
    
    train_loader = create_dataset(
        text=train_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=0,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="TREINAMENTO"
    )

    test_loader = create_dataset(
        text=test_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=0,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="TESTE"
    )

    val_loader = create_dataset(
        text=val_data,
        max_length=max_length,
        stride=1,
        batch_size=batch_sz,
        num_workers=0,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="VALIDAÇÃO"
    )
    
    return train_loader, test_loader, val_loader