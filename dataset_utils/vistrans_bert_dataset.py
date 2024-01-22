import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class VisTrans_BERT_Dataset(Dataset):
    def __init__(
        self,
        data,
        classes_to_idx,
        image_preprocessor,
        text_tokenizer,
        device,
        max_seq_length=20,
        image_root="dataset/val2014-resised"):
        
        # data is a list of dict contain 3 keys {'image_path': ,'question':, 'answer': }
        self.data = data
        self.image_root = image_root
        self.classes_to_idx = classes_to_idx
        self.max_seq_length = max_seq_length
        self.image_preprocessor = image_preprocessor
        self.text_tokenizer = text_tokenizer
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.data[index].get('image_path'))
        image = Image.open(img_path)
        
        if self.img_feature_extractor:
            img = self.image_preprocessor(images=image, return_tensors="pt")
            img = img.pixel_values.squeeze(0).to(self.device)
            
        question = self.data[index].get('question')
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt"
                )
            
            question = question.input_ids.squeeze(0).to(self.device)
            
        label = self.data[index].get('answer')
        label = torch.tensor(
            self.classes_to_idx[label],
            dtype=torch.long,
        ).to(self.device)
        
        return img, question, label