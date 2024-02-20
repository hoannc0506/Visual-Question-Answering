import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class CNN_LSTM_Dataset(Dataset):
    def __init__(
        self,
        data,
        classes_to_idx,
        text_tokenizer,
        max_seq_len=30,
        transform=None,
        image_root="dataset/val2014-resised"):
        
        # data is a list of dict contain 3 keys {'image_path': ,'question':, 'answer': }
        self.data = data
        self.transform = transform
        self.image_root = image_root
        self.classes_to_idx = classes_to_idx
        self.max_seq_len = max_seq_len
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.data[index].get('image_path'))
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(image)
            
        question = self.data[index].get('question')
        question = self.text_tokenizer(question)
        question = torch.tensor(question, dtype=torch.long)
        
        label = self.data[index].get('answer')
        label = torch.tensor(
            self.classes_to_idx[label],
            dtype=torch.long,
        )
        
        return img, question, label

            
            