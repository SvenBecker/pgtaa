import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def reshaped_data(df: pd.DataFrame, window_size: int=100, pred_range: int=5, scaler=None):
    x = df.copy()
    y = df.iloc[:, :8]
    y = np.log(y.pct_change(pred_range) + 1)
    y = y.values[pred_range + 1:]
    x.iloc[:, :8] = x.iloc[:, :8].pct_change(1)
    x = x[1:-pred_range]
    try:
        # scaler has to be fitted already
        x = scaler.transform(x)
    except Exception as e:
        print(e)

    xdata, ydata = [], []
    for step in range(x.shape[0] - window_size):
        xdata.append(x[step: window_size + step])
        ydata.append(y[window_size + step: window_size + step + 1])

    xdata = np.array(xdata)
    ydata = np.array(ydata).reshape((x.shape[0] - window_size, y.shape[1]))

    # xdata has the shape (samples, window_size, columns)
    # ydata has the shape (samples, nb_assets)
    return xdata, ydata


class TimeSeriesDataset(Dataset):
    def __init__(self, xdata, ydata):
        super(TimeSeriesDataset, self).__init__()
        self.xdata = xdata
        self.ydata = ydata

    def __getitem__(self, x):
        return torch.from_numpy(self.xdata[x]), torch.from_numpy(self.ydata[x])

    def __len__(self):
        return len(self.ydata)

    @classmethod
    def from_spec(cls, df: pd.DataFrame, pred_range: int=5):
        from pgtaa import config as cfg
        xdata, ydata = reshaped_data(
            df=df,
            window_size=cfg.WINDOW_SIZE,
            pred_range=pred_range,
            scaler=cfg.get_scaler()
        )
        return cls(xdata, ydata)


def dl_from_spec(split: float=0.8, batch_size: int=16, shuffle: bool=True,
                num_workers: int=2, pred_range: int=5):
    """Creates two PyTorch dataloaders from config spec

    Keyword Arguments:
        split {float} -- train-valid splitsize (default: {0.8})
        batch_size {int} -- batch size for training (default: {32})
        shuffle {bool} -- shuffle samples after each epoch (default: {True})
        num_workers {int} -- number of processes (default: {2})
        pred_range {int} -- prediction horizon (default: {5})

    Returns:
        tuple -- two dataloader (train, valid)
    """

    from pgtaa.core.utils import read_data
    from pgtaa import config as cfg
    x = read_data(cfg.TRAIN_CSV, cfg.NB_ASSETS, return_array=False)
    train = x.iloc[:int(len(x)*split)]
    valid = x.iloc[int(len(x)*split):]

    ds_train = TimeSeriesDataset.from_spec(train, pred_range=pred_range)
    ds_valid = TimeSeriesDataset.from_spec(valid, pred_range=pred_range)

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers, drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=1,
                        shuffle=False, num_workers=1, drop_last=True)

    return dl_train, dl_valid


def fit(model, criterion, optimizer, batch_size: int=64, epochs: int=30, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    dl_train, dl_valid = dl_from_spec(batch_size=batch_size, num_workers=4)
    loss_progress = {"Train Loss": [], "Validation Loss": []}
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        pbar = tqdm(dl_train, file=sys.stdout)
        for i, data in enumerate(pbar):
            pbar.set_description("Epoch %s" % str(epoch + 1))
            inputs, label = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.view(batch_size,-1))
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # validation
        with torch.set_grad_enabled(False):
            for j, vdata in enumerate(dl_valid):
                vinput, vlabel = vdata[0].to(device), vdata[1].to(device)
                outputs = model(vinput.view(1, -1))
                loss = criterion(outputs, vlabel)
                val_loss += loss.item()

        print(f"Train Loss: {train_loss / (i + 1)}  Validation Loss: {val_loss / (j + 1)}\n")

    print("Finished training")
