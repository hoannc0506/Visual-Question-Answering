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


test_data = utils.load_dataset('dataset/vaq2.0.TestImages.txt')