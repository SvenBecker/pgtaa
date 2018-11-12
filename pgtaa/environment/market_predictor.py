import pandas as pd  
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.manifold import TSNE
from sklearn.externals import joblib

from pgtaa.core.predictor_preproc import dl_from_spec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

max_epochs = 5
train_dl, valid_dl = dl_from_spec()

# LSTM network
net = nn.Sequential(
    nn.LSTM(3200, 128, bias=True),
    nn.Dropout(p=0.4, inplace=True),
    nn.ReLU(inplace=True),
    nn.LSTM(128, 64, bias=True),
    nn.Dropout(p=0.4, inplace=True),
    nn.ReLU(inplace=True),
    nn.LSTM(64, 10, bias=True)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lr=1e-3)   


for epoch in range(max_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for i, data in enumerate(train_dl):
        input, label = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # validation
    with torch.set_grad_enabled(False):
        for j, vdata in enumerate(valid_dl):
            vinput, vlabel = vdata[0].to(device), vdata[1].to(device)
            outputs = net(vinput)
            loss = criterion(outputs, vlabel)
            val_loss += loss.item()

    print(f"Train Loss: {train_loss / (i + 1)}, Validation Loss: {val_loss / (j + 1)}")
    train_loss = 0.0
    val_loss = 0.0

print('Finished Training')
