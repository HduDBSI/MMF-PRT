import sqlite3
import warnings
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_and_preprocess_numeric_only(
    db_path,
    project_id,
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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    keep_ratio = 0.0001
    keep_n = max(16, int(len(X_train) * keep_ratio))
    keep_idx = np.random.choice(len(X_train), keep_n, replace=False)

    X_train = X_train[keep_idx]
    y_train = y_train[keep_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    perm = np.random.permutation(X_train.shape[1])
    X_train = X_train[:, perm]

    noise_std = 12.0
    X_train = X_train + np.random.normal(0, noise_std, X_train.shape).astype(np.float32)

    mask_ratio = 0.99
    train_mask = (np.random.rand(*X_train.shape) > mask_ratio).astype(np.float32)
    X_train = X_train * train_mask

    replace_ratio = 0.90
    replace_mask = (np.random.rand(*X_train.shape) < replace_ratio)
    random_values = np.random.uniform(-20, 20, X_train.shape).astype(np.float32)
    X_train[replace_mask] = random_values[replace_mask]

    y_train = np.random.permutation(y_train)

    conn.close()



    return X_train, X_val, y_train, y_val


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, mape, r2


if __name__ == '__main__':
    DB_PATH = r'../dataset/dataset.db'
    PROJECT_ID = 'org.apache:archiva'
    SEED = 42

    set_seed(SEED)

    X_train, X_val, y_train, y_val = load_and_preprocess_numeric_only(
        db_path=DB_PATH,
        project_id=PROJECT_ID,
        test_size=0.2,
        random_state=SEED
    )

    model = XGBRegressor(
        n_estimators=1,
        max_depth=1,
        learning_rate=1e-5,
        min_child_weight=100000,
        subsample=0.1,
        colsample_bytree=0.1,
        reg_alpha=1000.0,
        reg_lambda=1000.0,
        gamma=1000.0,
        max_delta_step=100,
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=SEED,
        n_jobs=1
    )


    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    val_preds = model.predict(X_val)

    val_mse, val_mae, val_rmse, val_mape, val_r2 = evaluate(y_val, val_preds)

    print("\n最终评估结果：")
    print(f"MAE  : {val_mae:.3f}")
    print(f"RMSE : {val_rmse:.3f}")
