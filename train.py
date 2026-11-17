import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

DATA = os.environ.get('MOA_DATA', r'c:\Users\asus\Desktop\CS4486\Projects\moa\data')


def load():
    tr = pd.read_csv(os.path.join(DATA, 'train_features.csv'))
    tg = pd.read_csv(os.path.join(DATA, 'train_targets_scored.csv'))
    te = pd.read_csv(os.path.join(DATA, 'test_features.csv'))

    mask = tr['cp_type'] != 'ctl_vehicle'
    tr = tr[mask].reset_index(drop=True)
    tg = tg[mask].reset_index(drop=True)

    target_cols = [c for c in tg.columns if c != 'sig_id']
    feat_cols = [c for c in tr.columns if c not in ('sig_id', 'cp_type')]

    def encode(df):
        out = df[feat_cols].copy()
        out['cp_time'] = df['cp_time'].astype('category').cat.codes
        out['cp_dose'] = df['cp_dose'].astype('category').cat.codes
        return out

    time_map = {'24': 0, '48': 1, '72': 2}
    dose_map = {'D1': 0, 'D2': 1}

    def encode2(df):
        out = df[feat_cols].copy()
        out['cp_time'] = df['cp_time'].astype(str).map(time_map).fillna(0).astype(int)
        out['cp_dose'] = df['cp_dose'].astype(str).map(dose_map).fillna(0).astype(int)
        return out

    X = encode2(tr).values.astype(np.float32)
    Xt = encode2(te).values.astype(np.float32)
    Y = tg[target_cols].values.astype(np.float32)
    return X, Y, Xt, target_cols, te['sig_id'].values


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=1024, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_fold(Xtr, Ytr, Xva, Yva, Xte, epochs, bs, lr, seed, device,
               hidden=1024, smooth=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()

    xtr = torch.tensor(Xtr, device=device)
    ytr = torch.tensor(Ytr, device=device)
    if smooth > 0:
        ytr = ytr * (1 - smooth) + 0.5 * smooth
    xva = torch.tensor(Xva, device=device)
    xte = torch.tensor(Xte, device=device)

    n = xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            out = model(xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(xva)).cpu().numpy()
        pte = torch.sigmoid(model(xte)).cpu().numpy()
    return pva, pte


def score(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    vals = []
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0 or y[:, j].sum() == y.shape[0]:
            continue
        vals.append(log_loss(y[:, j], p[:, j], labels=[0, 1]))
    return float(np.mean(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model_seed', type=int, default=42)
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--smooth', type=float, default=0.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=r'c:\Users\asus\Desktop\CS4486\Projects\moa\results')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    X, Y, Xt, target_cols, test_ids = load()
    print('train', X.shape, 'targets', Y.shape, 'test', Xt.shape)
    print('missing rate %.5f' % np.isnan(X).mean())
    print('mean positives per target %.1f' % Y.sum(0).mean())

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros_like(Y)
    test_pred = np.zeros((Xt.shape[0], Y.shape[1]), dtype=np.float32)
    fold_scores = []

    for f, (ti, vi) in enumerate(kf.split(X)):
        t0 = time.time()
        pva, pte = train_fold(X[ti], Y[ti], X[vi], Y[vi], Xt,
                              args.epochs, args.bs, args.lr,
                              args.model_seed + f, device,
                              hidden=args.hidden, smooth=args.smooth)
        oof[vi] = pva
        test_pred += pte / args.folds
        s, k = score(Y[vi], pva)
        fold_scores.append(s)
        print('fold %d  logloss %.5f  (%d cols, %.0fs)' % (f, s, k, time.time() - t0))

    overall, k = score(Y, oof)
    print('OOF mean column-wise log loss %.5f (%d cols)' % (overall, k))

    res = dict(oof_logloss=overall, fold_scores=fold_scores,
               n_train=int(X.shape[0]), n_test=int(Xt.shape[0]),
               n_features=int(X.shape[1]), n_targets=int(Y.shape[1]),
               missing_rate=float(np.isnan(X).mean()),
               mean_positives=float(Y.sum(0).mean()),
               epochs=args.epochs, batch=args.bs, lr=args.lr,
               folds=args.folds, seed=args.seed, device=device,
               hidden=args.hidden, smooth=args.smooth, tag=args.tag)
    tag = ('_' + args.tag) if args.tag else ''
    with open(os.path.join(args.out, 'results%s.json' % tag), 'w') as fh:
        json.dump(res, fh, indent=2)

    np.save(os.path.join(args.out, 'oof%s.npy' % tag), oof)
    np.save(os.path.join(args.out, 'test_pred%s.npy' % tag), test_pred)
    with open(os.path.join(args.out, 'targets.json'), 'w') as fh:
        json.dump(target_cols, fh)
    pd.Series(test_pred[:, 0]).to_csv(os.path.join(args.out, 'dummy.csv'), index=False)


if __name__ == '__main__':
    main()
