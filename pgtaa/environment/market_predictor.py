import pandas as pd  
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

#from sklearn.svm import SVR
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import SGDRegressor
#from sklearn.manifold import TSNE
#from sklearn.externals import joblib

from pgtaa.core.predictor_preproc import dl_from_spec

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

max_epochs = 2
dl_train, dl_valid = dl_from_spec(batch_size=1)

# Linear network
class PredDense(nn.Module):
    def __init__(self, input_dim=3200, hidden_dim=(512, 128, 32), batch_size=1, output_dim=8):
        super(PredDense, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim[0])
        self.lin2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.lin3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.lin = nn.Linear(hidden_dim[2], output_dim, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.6, inplace=False)
        self.batch_size = batch_size
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin(x)
        return x
    
net = PredDense().double().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)   


for epoch in range(max_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for i, data in tqdm(enumerate(dl_train)):
        inputs, label = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # (seq_length,1,input_dim)
        #x = x.view(len(x), self.batch_size, -1)
        outputs = net(inputs.view(1,-1))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # validation
    with torch.set_grad_enabled(False):
        for j, vdata in enumerate(dl_valid):
            vinput, vlabel = vdata[0].to(device), vdata[1].to(device)
            outputs = net(vinput.view(1, -1))
            loss = criterion(outputs, vlabel)
            val_loss += loss.item()
    print(f"Train Loss: {train_loss / (i + 1)}, Validation Loss: {val_loss / (j + 1)}")
    train_loss = 0.0
    val_loss = 0.0

print('Finished Training')