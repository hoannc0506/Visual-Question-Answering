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



class VisTrans_BERT_Dataset(Dataset):
    def __init__(
        self,
        data,
        classes_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        image_root="dataset/val2014-resised"):
        
        # data is a list of dict contain 3 keys {'image_path': ,'question':, 'answer': }
        self.data = data
        self.image_root = image_root
        self.classes_to_idx = classes_to_idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.data[index].get('image_path'))
        image = Image.open(img_path)
        
        if self.img_feature_extractor:
            img = self.img_feature_extractor(images=image, return_tensors="pt")
            img = {k:v.to(self.device).squeeze(0) for k, v in img.items()}
            
        question = self.data[index].get('question')
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_tensors="pt"
                )
            
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}
            
        label = self.data[index].get('answer')
        label = torch.tensor(
            self.classes_to_idx[label],
            dtype=torch.long,
        ).to(self.device)
        
        sample = {'images': img,
                  'question': question,
                  'label': label}
        
        return sample
            
            