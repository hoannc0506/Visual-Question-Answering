import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt

import utils
from torch.utils.data import DataLoader
from dataset_utils.vistrans_bert_dataset import VisTrans_BERT_Dataset

from models.vistrans_bert import VisTrans_RoBERTa
from transformers import ViTImageProcessor
from transformers import AutoTokenizer

import trainer as trainer



# load data
train_data = utils.load_dataset('dataset/vaq2.0.TrainImages.txt')
val_data = utils.load_dataset('dataset/vaq2.0.DevImages.txt')
test_data = utils.load_dataset('dataset/vaq2.0.TestImages.txt')

print(len(train_data), len(val_data), len(test_data))

# create mapping dict
classes = set([sample['answer'] for sample in train_data])
classes_to_idx = {cls_name:idx for idx, cls_name in enumerate(classes)}
idx_to_classes = {idx:cls_name for idx, cls_name in enumerate(classes)}

# create tokenizer, preprocessor
text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
device = 'cuda:2'

# create datasets
train_dataset = VisTrans_BERT_Dataset(train_data, classes_to_idx, image_preprocessor, text_tokenizer)
val_dataset = VisTrans_BERT_Dataset(val_data, classes_to_idx, image_preprocessor, text_tokenizer)
test_dataset = VisTrans_BERT_Dataset(test_data, classes_to_idx, image_preprocessor, text_tokenizer)

# create dataloaders
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# sample
image, question, label = next(iter(train_loader))
print(image.shape, question.shape, label.shape)

# exit()

# define models
n_classes = len(classes)
hidden_size = 1024
n_layers = 1

model = VisTrans_RoBERTa(hidden_size=hidden_size,
                         n_layers=n_layers,
                         n_classes=n_classes)

model = model.to(device)
model.freeze()

lr = 1e-2
epochs = 50
weight_decay = 1e-5
scheduler_step_size = epochs *0.6
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

# train model
train_losses, val_losses = trainer.fit(model, train_loader, val_loader, criterion, 
                                         optimizer, scheduler, device, epochs)

df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
df.to_csv("logs/vistrans_roberta_results.csv", index=False)

val_loss, val_acc = trainer.evaluate(model, val_loader, criterion, device)

test_loss, test_acc = trainer.evaluate(model, val_loader, criterion, device)

print("Val acc", val_acc)
print("Test acc", test_acc)