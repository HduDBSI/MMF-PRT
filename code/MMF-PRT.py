# 三模态技术债务剩余寿命(版本数)回归预测（数值分支 + 文本分支 + 图分支（端到端 GAT））
# 多模态数据融合的技术债务消除时机预测方法
# 融合方式：跨模态注意力（Cross Attention）
# 说明：图是基于 commit 共现构建的文件共变图（file co-change graph），图分支使用 PyG 的 GATConv

import sqlite3
import itertools
import re
import warnings
import os
import numpy as np
import pandas as pd
import networkx as nx

from concurrent.futures import ThreadPoolExecutor

# sklearn 相关导入（回归指标）
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch 及工具
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.amp

# 句向量工具
from sentence_transformers import SentenceTransformer

# PyG：使用 GATConv 实现图注意力网络
try:
    from torch_geometric.nn import GATConv
except Exception as e:
    raise ImportError("未能导入 torch_geometric，请先安装 PyTorch-Geometric，参见 https://pytorch-geometric.readthedocs.io/ 。")

warnings.filterwarnings("ignore")

# ---------------- 文本清洗函数 ----------------
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\.,;:\-_/\\|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- 数据加载与预处理----------------
def load_and_preprocess(db_path, project_id='org.apache:zookeeper', test_size=0.2, random_state=42, embedding_cache_dir='embedding_cache'):
    os.makedirs(embedding_cache_dir, exist_ok=True)
    embedding_cache_path = os.path.join(embedding_cache_dir, f"{project_id.replace(':','_')}_text_embeddings.npy")
    graph_cache_path = os.path.join(embedding_cache_dir, f"{project_id.replace(':','_')}_graph_features.npz")

    conn = sqlite3.connect(db_path)
    sql = f'''
    SELECT * 
    FROM TD_FEATURES 
    WHERE 
    PROJECT_ID = '{project_id}'; 
    '''
    df = pd.read_sql_query(sql, conn)
    print(f"项目: {project_id}")

    # ------------------- 构建提交文件列表用于共变图 -------------------
    commit_files_sql = f'''
    SELECT * 
        FROM ISSUE_COMPONENTS 
        WHERE 
            PROJECT_ID = '{project_id}';
    '''
    commit_files = pd.read_sql_query(commit_files_sql, conn)
    commit_files['file_list'] = commit_files['components'].fillna('').apply(lambda s: [x for x in s.split('||') if x])

    # ------------------- 图特征加载或构建 -------------------
    if os.path.exists(graph_cache_path):
        print(f"加载已缓存图特征: {graph_cache_path}")
        npz = np.load(graph_cache_path, allow_pickle=True)
        node_feature_matrix = npz['node_feature_matrix']
        edge_index = npz['edge_index']
        components_unique = npz['components_unique'].tolist()
        df_node_feats = pd.DataFrame({
            'component': components_unique,
            'g_degree': node_feature_matrix[:, 0],
            'g_wdeg': node_feature_matrix[:, 1],
            'g_pagerank': node_feature_matrix[:, 2],
            'g_clustering': node_feature_matrix[:, 3],
        })
        df = df.merge(df_node_feats, on='component', how='left')
        df[['g_degree', 'g_wdeg', 'g_pagerank', 'g_clustering']] = df[['g_degree', 'g_wdeg', 'g_pagerank', 'g_clustering']].fillna(0.0)
    else:
        print(f"图特征缓存中: {graph_cache_path}")
        G = nx.Graph()
        for files in commit_files['file_list']:
            files = list(set(files))
            for f in files:
                if not G.has_node(f):
                    G.add_node(f)
            for u, v in itertools.combinations(files, 2):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
        print(f"共变图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

        num_nodes = G.number_of_nodes()
        try:
            pagerank_dict = nx.pagerank(G, weight='weight') if num_nodes > 0 else {}
        except Exception:
            pagerank_dict = {n: 0.0 for n in G.nodes()}
        clustering_dict = nx.clustering(G, weight='weight') if num_nodes > 0 else {}

        node_feats = []
        for node in G.nodes():
            deg = G.degree(node)
            wdeg = sum(d.get('weight', 1) for _, _, d in G.edges(node, data=True))
            pr = pagerank_dict.get(node, 0.0)
            clus = clustering_dict.get(node, 0.0)
            node_feats.append({'component': node, 'g_degree': deg, 'g_wdeg': wdeg, 'g_pagerank': pr, 'g_clustering': clus})
        df_node_feats = pd.DataFrame(node_feats)
        df = df.merge(df_node_feats, left_on='component', right_on='component', how='left')
        df[['g_degree', 'g_wdeg', 'g_pagerank', 'g_clustering']] = df[['g_degree', 'g_wdeg', 'g_pagerank', 'g_clustering']].fillna(0.0)

        components_unique = list(df_node_feats['component'].unique())
        node_feature_list = []
        for comp in components_unique:
            row = df_node_feats.loc[df_node_feats['component'] == comp]
            if not row.empty:
                node_feature_list.append([
                    float(row['g_degree'].values[0]),
                    float(row['g_wdeg'].values[0]),
                    float(row['g_pagerank'].values[0]),
                    float(row['g_clustering'].values[0])
                ])
            else:
                node_feature_list.append([0.0, 0.0, 0.0, 0.0])
        node_feature_matrix = np.array(node_feature_list, dtype=np.float32)

        edge_list = []
        for u, v, d in G.edges(data=True):
            if u in components_unique and v in components_unique:
                ui = components_unique.index(u)
                vi = components_unique.index(v)
                edge_list.append((ui, vi))
                edge_list.append((vi, ui))
        edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.array([[0],[0]], dtype=np.int64)

        np.savez(graph_cache_path, node_feature_matrix=node_feature_matrix, edge_index=edge_index, components_unique=components_unique)
        print(f"图特征缓存完成: {graph_cache_path}")

    # ------------------- 数值特征 -------------------
    feature_cols = [
        'LOC', 'CLOC', 'NM', 'NOC', 'CC', 'CCL', 'WMC', 'DLOC', 'CCR', 'DLOB', 'CBO', 'RFC', 'IC', 'CBM', 
        'COUNT', 'TYPE', 'SEVERITY', 'EFFORT'
    ]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    # versions 作为目标，确保为数值
    df['versions'] = pd.to_numeric(df['versions'], errors='coerce')
    # 丢弃数值特征或目标为 NA 的样本
    df = df.dropna(subset=feature_cols + ['versions'])

    # ------------------- 文本嵌入 -------------------
    df['MESSAGE_COMPONENT_COMBINED'] = df['MESSAGE_COMPONENT_COMBINED'].apply(clean_text)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(embedding_cache_path):
        print(f"加载已缓存文本嵌入: {embedding_cache_path}")
        X_text = np.load(embedding_cache_path)
    else:
        print(f"生成文本嵌入并缓存至 {embedding_cache_path} ... 设备: {DEVICE}")
        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        messages = df['MESSAGE_COMPONENT_COMBINED'].fillna('').astype(str).tolist()
        text_embeddings = text_model.encode(messages, show_progress_bar=True)
        X_text = np.array(text_embeddings, dtype=np.float32)
        np.save(embedding_cache_path, X_text)

    # ------------------- 标准化数值特征 -------------------
    X_num = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    # ------------------- node_idx -------------------
    node_to_idx = {comp: idx for idx, comp in enumerate(components_unique)}
    sample_node_idx = [node_to_idx.setdefault(comp, len(node_to_idx)) for comp in df['component'].values]
    node_idx_arr = np.array(sample_node_idx, dtype=np.int64)

    # ------------------- 划分训练/验证集（目标为 versions） -------------------
    y = df['versions'].astype(np.float32).values
    # y = np.log1p(df['versions'].astype(np.float32).values)  # log1p 变换
    keys = df['ISSUE_KEY'].values
    (X_num_train, X_num_val,
     X_text_train, X_text_val,
     node_idx_train, node_idx_val,
     y_train, y_val,
     key_train, key_val) = train_test_split(
        X_num, X_text, node_idx_arr, y, keys,
        test_size=test_size, random_state=random_state
    )

    conn.close()
    return (X_num_train, X_num_val, X_text_train, X_text_val,
            node_idx_train, node_idx_val, y_train, y_val,
            key_train, key_val, scaler,
            edge_index, node_feature_matrix)

# ---------------- Dataset 与 DataLoader（支持 node_idx） ----------------
class DebtDataset(Dataset):
    """三模态 Dataset：返回 (X_num, X_text, node_idx, y, [key]) - y 为 float（回归）"""
    def __init__(self, X_num, X_text, node_idx, y, keys=None):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_text = torch.from_numpy(X_text).float()
        self.node_idx = torch.from_numpy(node_idx).long()
        self.y = torch.from_numpy(y).float()  # 回归目标 float
        self.keys = keys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.keys is not None:
            return (self.X_num[idx], self.X_text[idx], self.node_idx[idx], self.y[idx], self.keys[idx])
        return (self.X_num[idx], self.X_text[idx], self.node_idx[idx], self.y[idx])

def create_data_loaders(X_num_train, X_text_train, node_idx_train, y_train,
                        X_num_val, X_text_val, node_idx_val, y_val,
                        batch_size=128):
    """回归不使用加权采样器，训练集 shuffle=True"""
    train_ds = DebtDataset(X_num_train, X_text_train, node_idx_train, y_train)
    val_ds = DebtDataset(X_num_val, X_text_val, node_idx_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader

# ---------------- 数值分支注意力模块 ----------------
class FeatureAttention(nn.Module):
    def __init__(self, num_feats, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feat_embed = nn.Sequential(nn.Linear(num_feats, d_model), nn.GELU())
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        v = self.feat_embed(x)
        B = x.size(0)
        Q = self.W_q(v).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(v).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(v).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, device=x.device, dtype=torch.float32))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = (attn_weights @ V).permute(0, 2, 1, 3).reshape(B, -1, self.d_model)
        output = self.norm(self.alpha * output + v.unsqueeze(1))
        return output.mean(dim=1), attn_weights.mean(dim=1)

# ---------------- 跨模态注意力模块 ----------------
class CrossAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        Q = self.W_q(query).view(B, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_k(key).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_v(value).view(B, Tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, device=query.device, dtype=torch.float32))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).permute(0, 2, 1, 3).reshape(B, Tq, self.d_model)
        out = self.out(attn_output)
        out = self.norm(out + query)
        return out.squeeze(1), attn_weights

# ---------------- 图分支（端到端 GAT）与三模态回归模型主体 ----------------
class DebtLifetimeRegressorGNNTriModal(nn.Module):
    """
    三模态回归模型（使用 GATConv）：
      - 数值分支：FeatureAttention
      - 文本分支：Linear -> GELU -> Dropout
      - 图分支：GATConv 两层
      - 跨模态注意力：三模态两两交互
      - 融合 MLP 输出单个回归值
    """
    def __init__(self, num_numeric_feats, text_embed_dim=384, graph_feat_dim=4, d_model=128, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.num_attn = FeatureAttention(num_numeric_feats, d_model=d_model, num_heads=num_heads)
        self.text_processor = nn.Sequential(nn.Linear(text_embed_dim, d_model), nn.GELU(), nn.Dropout(0.2))

        self.gat1 = GATConv(graph_feat_dim, d_model // num_heads, heads=num_heads, concat=True, dropout=0.2)
        self.gat_bn1 = nn.BatchNorm1d(d_model)
        self.gat2 = GATConv(d_model, d_model // num_heads, heads=num_heads, concat=True, dropout=0.2)
        self.gat_bn2 = nn.BatchNorm1d(d_model)

        self.cross_num = CrossAttention(d_model=d_model, num_heads=num_heads)
        self.cross_text = CrossAttention(d_model=d_model, num_heads=num_heads)
        self.cross_graph = CrossAttention(d_model=d_model, num_heads=num_heads)

        # 融合 MLP -> 单输出回归
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 回归单输出
        )

        self.register_buffer('edge_index_buffer', torch.zeros((2, 1), dtype=torch.long))
        self._node_feat_buffer = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_graph(self, edge_index, node_feature_matrix):
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()
        self.edge_index_buffer = edge_index.long().contiguous()
        if isinstance(node_feature_matrix, np.ndarray):
            node_feature_matrix = torch.from_numpy(node_feature_matrix).float()
        self._node_feat_buffer = node_feature_matrix

    def compute_node_embeddings(self, device):
        x = self._node_feat_buffer.to(device)
        edge_index = self.edge_index_buffer.to(device)
        x = self.gat1(x, edge_index)
        x = self.gat_bn1(x)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = self.gat_bn2(x)
        x = torch.relu(x)
        return x

    def forward(self, num_input, text_input, node_idx_batch):
        num_feat, num_attn_map = self.num_attn(num_input)
        text_feat = self.text_processor(text_input)
        device = num_input.device
        node_emb = self.compute_node_embeddings(device)
        graph_feat = node_emb[node_idx_batch]

        tv = torch.cat([text_feat.unsqueeze(1), graph_feat.unsqueeze(1)], dim=1)
        num_cross, attn_num = self.cross_num(num_feat, tv, tv)

        nv = torch.cat([num_feat.unsqueeze(1), graph_feat.unsqueeze(1)], dim=1)
        text_cross, attn_text = self.cross_text(text_feat, nv, nv)

        nt = torch.cat([num_feat.unsqueeze(1), text_feat.unsqueeze(1)], dim=1)
        graph_cross, attn_graph = self.cross_graph(graph_feat, nt, nt)

        combined = torch.cat([num_cross, text_cross, graph_cross], dim=1)
        out = self.fusion(combined)  # [B, 1]
        return out.squeeze(1), (attn_num, attn_text, attn_graph)  # 返回 [B] 预测值

# ---------------- 训练与评估函数（回归） ----------------
def train_epoch(model, loader, criterion, optimizer, device, grad_clip=2.0):
    model.train()
    total_loss, total_samples = 0.0, 0
    scaler = torch.amp.GradScaler()

    for batch in loader:
        if len(batch) == 5:
            num_batch, text_batch, node_idx_batch, y_batch, _ = batch
        else:
            num_batch, text_batch, node_idx_batch, y_batch = batch

        num_batch = num_batch.to(device)
        text_batch = text_batch.to(device)
        node_idx_batch = node_idx_batch.to(device)
        y_batch = y_batch.to(device)

        batch_size = num_batch.size(0)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            preds, _ = model(num_batch, text_batch, node_idx_batch)  # [B]
            loss = criterion(preds, y_batch)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                num_batch, text_batch, node_idx_batch, y_batch, _ = batch
            else:
                num_batch, text_batch, node_idx_batch, y_batch = batch

            num_batch = num_batch.to(device)
            text_batch = text_batch.to(device)
            node_idx_batch = node_idx_batch.to(device)
            y_batch = y_batch.to(device)

            batch_size = num_batch.size(0)
            preds, _ = model(num_batch, text_batch, node_idx_batch)  # [B]
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y_batch.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100
    r2 = r2_score(all_labels, all_preds)

    return (total_loss / total_samples, mse, mae, rmse, mape, r2, all_preds, all_labels)


# ---------------- 主程序入口 ----------------
if __name__ == '__main__':
    DB_PATH = '../dataset/dataset.db'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"运行设备: {DEVICE}")

    (X_num_train, X_num_val, X_text_train, X_text_val,
     node_idx_train, node_idx_val, y_train, y_val,
     key_train, key_val, scaler,
     edge_index, node_feature_matrix) = load_and_preprocess(DB_PATH)

    train_loader, val_loader = create_data_loaders(
        X_num_train, X_text_train, node_idx_train, y_train,
        X_num_val, X_text_val, node_idx_val, y_val,
        batch_size=128
    )

    model = DebtLifetimeRegressorGNNTriModal(
        num_numeric_feats=X_num_train.shape[1],
        text_embed_dim=X_text_train.shape[1],
        graph_feat_dim=node_feature_matrix.shape[1],
        d_model=128,
        num_heads=4
    ).to(DEVICE)

    model.set_graph(edge_index, node_feature_matrix)

    # 回归损失（MSE），如需 MAE 可改为 nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
    # 监控 val_mae（或 val_mse），ReduceLROnPlateau mode='min'
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_mae = float('inf')
    no_improve = 0

    print("\n开始训练（回归）...")

    for epoch in range(1, 201):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mse, val_mae, val_rmse, val_mape, val_r2, val_preds, val_labels = eval_epoch(model, val_loader, criterion, DEVICE)

        # 以 MAE 作为主监控指标（因为解释性好），也可换成 val_mse
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), 'best_debt_model_gat_trimodal_regressor.pth')
            print(f"保存最佳模型（MAE：{val_mae:.4f}）")
        else:
            no_improve += 1
            if no_improve >= 7:
                print("连续7轮未提升，提前终止训练")
                break

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | MAE: {val_mae:.3f} | RMSE: {val_rmse:.3f} | 学习率: {lr:.1e}")

    # 最终评估并打印
    val_loss, val_mse, val_mae, val_rmse, val_mape, val_r2, val_preds, val_labels = eval_epoch(model, val_loader, criterion, DEVICE)
    print("\n最终评估结果 :")
    print(f"MAE: {val_mae:.3f}")
    print(f"RMSE: {val_rmse:.3f}")

