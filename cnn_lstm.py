import torch
import torch.nn as nn
import torchvision.transforms as transforms

import os
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import utils
# from utils import eng as eng_spacy_model
from VQA_datasets import CNN_LSTM_Dataset
import VQA_models
import train_val
import torchinfo


# load data
train_data = utils.load_dataset('dataset/vaq2.0.TrainImages.txt')
val_data = utils.load_dataset('dataset/vaq2.0.DevImages.txt')
test_data = utils.load_dataset('dataset/vaq2.0.TestImages.txt')

print(len(train_data), len(val_data), len(test_data))

# build vocab
vocab = utils.build_vocab(train_data)
print(len(vocab))

# create mapping dict
classes = set([sample['answer'] for sample in train_data])
classes_to_idx = {cls_name:idx for idx, cls_name in enumerate(classes)}
idx_to_classes = {idx:cls_name for idx, cls_name in enumerate(classes)}

# create tokenizer
text_tokenizer = utils.Tokenizer(vocab, max_sequence_length=30)

# create dataset 
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

# create datasets
train_dataset = CNN_LSTM_Dataset(train_data, classes_to_idx, text_tokenizer, transform=transform)
val_dataset = CNN_LSTM_Dataset(val_data, classes_to_idx, text_tokenizer, transform=transform)
test_dataset = CNN_LSTM_Dataset(test_data, classes_to_idx, text_tokenizer, transform=transform)

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
img_model_name = 'resnet50'
hidden_size = 128
n_layers = 1
embedding_dim = 128

device = 'cuda:2'
model = VQA_models.Resnet_BiLSTM(n_classes=len(classes), 
                                 vocab_length=len(vocab),
                                 img_model_name='resnet50',
                                 embed_dim=128,
                                 lstm_num_layers=1,
                                 lstm_hidden_size=128,
                                 dropout_prob=0.2
                                 )
model = model.to(device)

lr = 1e-2
epochs = 50
weight_decay = 1e-5
scheduler_step_size = epochs *0.6
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

# train model
train_losses, val_losses = train_val.fit(model, train_loader, val_loader, criterion, 
                                         optimizer, scheduler, device, epochs)

df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
df.to_csv("logs/cnn_lstm_results.csv", index=False)

val_loss, val_acc = train_val.evaluate(model, val_loader, criterion, device)

test_loss, test_acc = train_val.evaluate(model, val_loader, criterion, device)

print("Val acc", val_acc)
print("Test acc", test_acc)