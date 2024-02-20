import torch
from torch.utils.data import Dataset
import os
from PIL import Image

def load_dataset(data_path):
    data = []
    lines = open(data_path, 'r').read().strip().split('\n')
    for line in lines:
        temp = line.split('\t')
        qa = temp[1].split('?')
        
        answer = qa[2] if len(qa) == 3 else qa[1]

        data_sample = {'image_path': temp[0][:-2],
                    'question': qa[0] + '?',
                    'answer': answer.strip()}
        
        data.append(data_sample)

    return data

class VQADatasetFromPretrained(Dataset):
    def __init__(
        self,
        data,
        classes_to_idx,
        image_preprocessor,
        text_tokenizer,
        max_seq_length=20,
        image_root="dataset/val2014-resised"):
        
        # data is a list of dict contain 3 keys {'image_path': ,'question':, 'answer': }
        self.data = data
        self.image_root = image_root
        self.classes_to_idx = classes_to_idx
        self.max_seq_length = max_seq_length
        self.image_preprocessor = image_preprocessor
        self.text_tokenizer = text_tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.data[index].get('image_path'))
        image = Image.open(img_path)
        
        if self.image_preprocessor:
            img = self.image_preprocessor(images=image, return_tensors="pt")
            img = img.pixel_values.squeeze(0)
            
        question = self.data[index].get('question')
        if self.text_tokenizer:
            question = self.text_tokenizer(
                question,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt"
                )
            
            question = question.input_ids.squeeze(0)
            
        label = self.data[index].get('answer')
        label = torch.tensor(
            self.classes_to_idx[label],
            dtype=torch.long,
        )
        
        return img, question, label