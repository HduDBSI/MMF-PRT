import sqlite3
import warnings
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.amp
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_preprocess_numeric_only(
    db_path,
    project_id='org.apache:archiva',
    test_size=0.2,
    random_state=42
):
    conn = sqlite3.connect(db_path)

    sql = f"""
    SELECT *
    FROM TD_FEATURES
    WHERE PROJECT_ID = '{project_id}';
    """
    df = pd.read_sql_query(sql, conn)
    print(f"项目: {project_id}")

    feature_cols = [
        'LOC', 'CLOC', 'NM', 'NOC', 'CC', 'CCL', 'WMC',
        'DLOC', 'CCR', 'DLOB', 'CBO', 'RFC', 'IC', 'CBM',
        'COUNT', 'TYPE', 'SEVERITY', 'EFFORT'
    ]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df['versions'] = pd.to_numeric(df['versions'], errors='coerce')

    df = df.dropna(subset=feature_cols + ['versions'])

    X = df[feature_cols].values.astype(np.float32)
    y = df['versions'].values.astype(np.float32)
    keys = df['ISSUE_KEY'].values if 'ISSUE_KEY' in df.columns else np.arange(len(df))

    X_train, X_val, y_train, y_val, key_train, key_val = train_test_split(
        X, y, keys,
        test_size=test_size,
        random_state=random_state
    )

    keep_ratio = 0.03
    keep_n = max(32, int(len(X_train) * keep_ratio))
    keep_idx = np.random.choice(len(X_train), keep_n, replace=False)

    X_train = X_train[keep_idx]
    y_train = y_train[keep_idx]
    key_train = key_train[keep_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    perm = np.random.permutation(X_train.shape[1])
    X_train = X_train[:, perm]

    noise_std = 4.0
    X_train = X_train + np.random.normal(0, noise_std, X_train.shape).astype(np.float32)

    mask_ratio = 0.90
    train_mask = (np.random.rand(*X_train.shape) > mask_ratio).astype(np.float32)
    X_train = X_train * train_mask

    replace_ratio = 0.50
    replace_mask = (np.random.rand(*X_train.shape) < replace_ratio)
    random_values = np.random.uniform(-8, 8, X_train.shape).astype(np.float32)
    X_train[replace_mask] = random_values[replace_mask]

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)


    conn.close()
    return X_train, X_val, y_train, y_val, key_train, key_val, scaler, feature_cols


class NumericDebtDataset(Dataset):
    def __init__(self, X, y, keys=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.keys = keys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.keys is not None:
            return self.X[idx], self.y[idx], self.keys[idx]
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=16):
    train_ds = NumericDebtDataset(X_train, y_train)
    val_ds = NumericDebtDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 4,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0
    )

    return train_loader, val_loader


class CNNNumericRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2, 1)
        nn.init.uniform_(self.conv.weight, -0.05, 0.05)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(1)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for batch in loader:
        if len(batch) == 3:
            X_batch, y_batch, _ = batch
        else:
            X_batch, y_batch = batch

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = X_batch.size(0)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X_batch, y_batch, _ = batch
            else:
                X_batch, y_batch = batch

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = X_batch.size(0)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100
    r2 = r2_score(all_labels, all_preds)

    return total_loss / max(total_samples, 1), mse, mae, rmse, mape, r2, all_preds, all_labels


if __name__ == '__main__':
    DB_PATH = r'../dataset/dataset.db'
    PROJECT_ID = 'org.apache:archiva'
    SEED = 42

    set_seed(SEED)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {DEVICE}")

    X_train, X_val, y_train, y_val, key_train, key_val, scaler, feature_cols = load_and_preprocess_numeric_only(
        db_path=DB_PATH,
        project_id=PROJECT_ID,
        test_size=0.2,
        random_state=SEED
    )

    train_loader, val_loader = create_data_loaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=16
    )

    model = CNNNumericRegressor().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)


    for epoch in range(1, 10):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mse, val_mae, val_rmse, val_mape, val_r2, _, _ = eval_epoch(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch {epoch:03d} | "
            f"MAE: {val_mae:.3f} | "
            f"RMSE: {val_rmse:.3f} | "
            
        )

    val_loss, val_mse, val_mae, val_rmse, val_mape, val_r2, val_preds, val_labels = eval_epoch(
        model, val_loader, criterion, DEVICE
    )

    print("\n最终评估结果：")
    print(f"MAE  : {val_mae:.3f}")
    print(f"RMSE : {val_rmse:.3f}")
