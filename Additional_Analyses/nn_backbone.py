"""
Neural backbone for the robustness run: a multilayer perceptron on the same
27 inputs as the tree backbones, one network per horizon.

Training: standardised inputs and target (train-split statistics), AdamW,
mini-batches of 256, early stopping on the tune-split RMSE (patience 20
epochs, best weights restored), trained on the GPU with fixed seeds and
deterministic kernels. Prediction runs on the CPU.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np          # noqa: E402
import torch                # noqa: E402
from torch import nn        # noqa: E402

import her_shift as hs      # noqa: E402

SEED = 42
MAX_EPOCHS = 300
PATIENCE = 20
BATCH = 256


class MLPRegressor:
    def __init__(self, hidden=(256, 256, 128), dropout=0.1, lr=1e-3,
                 weight_decay=1e-4, device="cuda"):
        self.hidden, self.dropout = hidden, dropout
        self.lr, self.wd = lr, weight_decay
        self.device = device if torch.cuda.is_available() else "cpu"

    def _net(self, d):
        layers, k = [], d
        for i, hdim in enumerate(self.hidden):
            layers += [nn.Linear(k, hdim), nn.GELU()]
            if i < len(self.hidden) - 1:
                layers.append(nn.Dropout(self.dropout))
            k = hdim
        layers.append(nn.Linear(k, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        X = np.asarray(X, dtype=np.float64)
        self.mx, self.sx = X.mean(0), X.std(0)
        self.sx[self.sx == 0] = 1.0
        y = np.asarray(y, dtype=np.float64)
        self.my, self.sy = y.mean(), y.std()
        dev = self.device
        Xt = torch.tensor((X - self.mx) / self.sx, dtype=torch.float32, device=dev)
        yt = torch.tensor((y - self.my) / self.sy, dtype=torch.float32, device=dev)
        Xv = torch.tensor((np.asarray(X_val, dtype=np.float64) - self.mx) / self.sx,
                          dtype=torch.float32, device=dev)
        yv = np.asarray(y_val, dtype=np.float64)
        net = self._net(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        g = torch.Generator(device=dev)
        g.manual_seed(SEED)
        best, best_state, wait = np.inf, None, 0
        n = len(Xt)
        for epoch in range(MAX_EPOCHS):
            net.train()
            perm = torch.randperm(n, generator=g, device=dev)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad(set_to_none=True)
                loss = torch.mean((net(Xt[b]).squeeze(1) - yt[b]) ** 2)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xv).squeeze(1).double().cpu().numpy() * self.sy + self.my
            rmse = float(np.sqrt(np.mean((yv - pv) ** 2)))
            if rmse < best - 1e-6:
                best, wait = rmse, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        self.epochs, self.tune_rmse = epoch + 1, best
        net = self._net(X.shape[1])
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        torch.use_deterministic_algorithms(False)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        with torch.no_grad():
            xt = torch.tensor((X - self.mx) / self.sx, dtype=torch.float32)
            out = self.net(xt).squeeze(1).double().numpy()
        return out * self.sy + self.my


def fit_mlp_backbone(train, tune):
    m = MLPRegressor().fit(train[hs.FEATURES], train["target"],
                           tune[hs.FEATURES], tune["target"])
    return f"MLP (epochs {m.epochs}, tune RMSE {m.tune_rmse:.4f})", m
