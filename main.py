import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

INPUT = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train_features.csv' in files:
        INPUT = root
        break
print('INPUT =', INPUT, flush=True)
print('input tree:', os.listdir('/kaggle/input'), flush=True)
if INPUT is None:
    for d in os.listdir('/kaggle/input'):
        print(d, os.listdir(os.path.join('/kaggle/input', d))[:8], flush=True)
EPOCHS = 25
BS = 128
LR = 1e-3
FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    def __init__(self, n_in, n_out, h=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(h, n_out),
        )

    def forward(self, x):
        return self.net(x)


def load():
    tr = pd.read_csv(os.path.join(INPUT, 'train_features.csv'))
    tg = pd.read_csv(os.path.join(INPUT, 'train_targets_scored.csv'))
    te = pd.read_csv(os.path.join(INPUT, 'test_features.csv'))
    sub = pd.read_csv(os.path.join(INPUT, 'sample_submission.csv'))

    mask = (tr['cp_type'] != 'ctl_vehicle').values
    tr = tr[mask].reset_index(drop=True)
    tg = tg[mask].reset_index(drop=True)

    cols = [c for c in tg.columns if c != 'sig_id']
    tmap = {24: 0, 48: 1, 72: 2}
    dmap = {'D1': 0, 'D2': 1}
    for df in (tr, te):
        df['cp_time'] = df['cp_time'].map(tmap).fillna(0).astype(int)
        df['cp_dose'] = df['cp_dose'].map(dmap).fillna(0).astype(int)

    drop = ['sig_id', 'cp_type']
    X = tr.drop(columns=drop).values.astype(np.float32)
    Xt = te.drop(columns=drop).values.astype(np.float32)
    Y = tg[cols].values.astype(np.float32)
    return X, Y, Xt, cols, te['sig_id'].values, sub


def run(X, Y, Xt, seed, smooth):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof = np.zeros_like(Y)
    pred = np.zeros((Xt.shape[0], Y.shape[1]), dtype=np.float32)
    xt = torch.tensor(Xt, device=DEVICE)

    for f, (ti, vi) in enumerate(kf.split(X)):
        torch.manual_seed(seed + f)
        np.random.seed(seed + f)
        model = MLP(X.shape[1], Y.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        lossf = nn.BCEWithLogitsLoss()

        xa = torch.tensor(X[ti], device=DEVICE)
        ya = torch.tensor(Y[ti], device=DEVICE)
        if smooth > 0:
            ya = ya * (1 - smooth) + 0.5 * smooth

        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(xa.shape[0], device=DEVICE)
            for i in range(0, xa.shape[0], BS):
                idx = perm[i:i + BS]
                opt.zero_grad()
                loss = lossf(model(xa[idx]), ya[idx])
                loss.backward()
                opt.step()
            sch.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X[vi], device=DEVICE)
            oof[vi] = torch.sigmoid(model(xv)).cpu().numpy()
            pred += torch.sigmoid(model(xt)).cpu().numpy() / FOLDS
        print('seed %d fold %d done' % (seed, f), flush=True)
    return oof, pred


X, Y, Xt, cols, sig_id, sub = load()
print('train', X.shape, 'test', Xt.shape, 'device', DEVICE, flush=True)

oof_a, pred_a = run(X, Y, Xt, 42, 0.001)
oof_b, pred_b = run(X, Y, Xt, 123, 0.0)

p = (np.clip(oof_a, 1e-6, 1 - 1e-6) + np.clip(oof_b, 1e-6, 1 - 1e-6)) / 2
ll = -(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean()
print('OOF log loss %.5f' % ll, flush=True)

final = (pred_a + pred_b) / 2
out = pd.DataFrame(final, columns=cols)
out.insert(0, 'sig_id', sig_id)
out = out[sub.columns.tolist()]
out.to_csv('submission.csv', index=False)
print('saved', out.shape, flush=True)
