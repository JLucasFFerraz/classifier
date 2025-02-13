from transformers import BertModel, BertTokenizer
import re
import torch
import torch.nn as nn
from main import plot_loss_train, dynamic_lr_stratified_train_test_val_fine_tuning, plot_loss_test
from models.protbert_classifiers.NN import PBLinearClassifier
import pickle
import random
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models.protbert_classifiers.NN import PBLinearClassifier
from models.protbert_classifiers.RandomForest import PBRandomForestClassifier
from torch.utils.data import DataLoader
import os
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit

import os
import pandas as pd
from torch.utils.data import DataLoader

import numpy as np
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

torch.cuda.empty_cache()
# Check for GPU availability and allow GPU use if available
print("Loading")
print("GPU available:", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.BCEWithLogitsLoss()#nn.CosineSimilarity(dim=0)
print("Example loss:")
zero_loss = criterion(torch.Tensor([1.0, 0.0]), torch.Tensor([1.0, 0.0]))
one_loss = criterion(torch.Tensor([1.0, 0.0]), torch.Tensor([0.0, 1.0]))
print("1 vs 0", one_loss.item())
print("0 vs 0", zero_loss.item())

all_data = []
for i in range(100):
    all_data.append(("[CLS] A A A [SEP]", 0))
    all_data.append(("[CLS] B B B [SEP]", 1))
random.shuffle(all_data)

bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
bert_model = bert_model.to(device)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = PBLinearClassifier(bert_model, tokenizer, device)
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.00001)
batch_size = 2
test_loader = DataLoader(all_data, batch_size=batch_size)
random.shuffle(all_data)
train_loader = DataLoader(all_data, batch_size=batch_size)


test_losses = []
train_losses = []
for e in range(5):
    avg_train_loss, every_train_loss = model.train(train_loader, optimizer, criterion)
    train_losses.append(avg_train_loss)
    
    test_stats, avg_test_loss, _ = model.test(test_loader, criterion)
    test_losses.append(avg_test_loss)
    plot_loss_train(train_losses, None, "./train.png")    
    plot_loss_test(test_losses, None, "./test.png")

