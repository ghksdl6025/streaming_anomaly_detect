import pandas as pd
from river import stream,tree,metrics
import utils
from encoding import prefix_bin
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os
from tqdm import tqdm
import sliding_window
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
import datetime, time
import importlib

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device('cuda')

import torch.optim as optim

importlib.reload(sliding_window)

model = nn.LSTM(input_size=3,hidden_size=6,num_layers=2)
h0 = torch.randn(2,1,6)
c0 = torch.randn(2,1,6)


input = torch.randn(40,1,3)
print(input)

output, (hn,cn) = model(input, (h0,c0))
print(output.shape)