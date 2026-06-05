import pandas as pd
import numpy as np
import os
import argparse
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--torch_seed", type=int, default=1337)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--low_threshold", type=float, default=0.96)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--date_class", type=int, default=0)

args, _ = parser.parse_known_args()
print("Arguments:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}")


############################# Cell 1 #############################
# =========================================================
# 1. Index list
# =========================================================
# Source: AKShare.
# The index data are obtained from the Chinese financial data platform AKShare,
# therefore the original column names and index descriptions are kept in Chinese.
indices = {
    'sp500': '标普500',
    'dow': '道琼斯',
    'nasdaq': '纳斯达克',
    'ftse100': '英国富时100',
    'dax': '德国DAX30',
    'cac40': '法国CAC40',
    'nikkei225': '日经225',
    'hangseng': '恒生指数',
    'sse50': '上证指数',
    'hs300': '沪深300'
}

train_val_indices = ['sp500','dow','dax','ftse100','cac40','nikkei225','hangseng','sse50','hs300','nasdaq']
test_indices = ['sp500','dow','dax','ftse100','cac40','nikkei225','hangseng','sse50','hs300','nasdaq']

csv_folder = "global_index"

# =========================================================
# 2. parameters
# =========================================================
window_size = 11
start_col = 2
price_col = '最新价' # "最新价" means "latest price" in Chinese, which is the column used for regime classification.

# regime threshold
LOW_THRESHOLD = args.low_threshold
HIGH_THRESHOLD = (1-LOW_THRESHOLD)+1

if args.date_class == 0:
    train_date_ranges = {
        'train': ("1999-01-01", "2008-12-31"),
        'val': ("2009-01-01", "2014-12-31"),
        'test': ("2015-01-09", "2019-12-31")
    }
    test_date_ranges = {
        'test': ("2015-01-09", "2019-12-31")
    }

else:
    train_date_ranges = {
        'train': ("2005-01-01", "2014-12-31"),
        'val': ("2015-01-01", "2020-12-31"),
        'test': ("2021-01-05", "2025-12-31")
    }

    test_date_ranges = {
        'test': ("2021-01-05", "2025-12-31")
    }

# =========================================================
# 3. time range conversion
# =========================================================
train_date_ranges = {
    k: (pd.to_datetime(v[0]), pd.to_datetime(v[1]))
    for k, v in train_date_ranges.items()
}

test_date_ranges = {
    k: (pd.to_datetime(v[0]), pd.to_datetime(v[1]))
    for k, v in test_date_ranges.items()
}

# =========================================================
# 4. Window Creation with Labels
# =========================================================
def create_windowed_data_with_labels(
    data,
    window_size=28,
    start_col=2,
    price_col='最新价' # "最新价" means "latest price" in Chinese, which is the column used for regime classification.
):

    features = data.iloc[:, start_col:].values

    # position of price column in the features
    price_index = data.columns.get_loc(price_col) - start_col

    X = []
    y = []
    dates = []

    for i in range(len(features) - window_size + 1):

        # =================================================
        # window
        # =================================================
        window = features[i:i+window_size].copy()

        # =================================================
        # normalization
        # =================================================
        baseline = window[-2].copy()

        baseline[baseline == 0] = 1e-10

        normalized_window = window / baseline

        # =================================================
        # label
        # =================================================
        last_price = normalized_window[-1, price_index]

        # 0 = strong down
        # 1 = strong up
        # 2 = neutral
        if last_price < LOW_THRESHOLD:

            label = 0

        elif last_price > HIGH_THRESHOLD:

            label = 1

        else:

            label = 2

        X.append(normalized_window)

        y.append(label)

        # last date in the window as the timestamp for this sample
        dates.append(
            data.index[i + window_size - 1]
        )

    return (
        np.array(X),
        np.array(y),
        np.array(dates)
    )

# =========================================================
# 5. init lists and dicts
# =========================================================
X_train_list = []
X_val_list = []

y_train_list = []
y_val_list = []

X_test_dict = {}
y_test_dict = {}

# =========================================================
# 6. process train/val indices
# =========================================================

for key in train_val_indices:

    symbol = indices[key]

    csv_path = f"{csv_folder}/{key}.csv"

    if not os.path.exists(csv_path):

        print("can't find:", csv_path)

        continue

    # =====================================================
    # read csv
    # =====================================================
    df = pd.read_csv(
        csv_path,
        encoding='utf-8-sig'
    )
    #日期 means "date" in Chinese. If the date column exists, convert it to datetime and set as index.
    if '日期' in df.columns:

        df['日期'] = pd.to_datetime(df['日期'])

        df.set_index('日期', inplace=True)

        df.sort_index(inplace=True)

    # =====================================================
    # create windows
    # =====================================================
    X_all, y_all, dates_all = (
        create_windowed_data_with_labels(
            df,
            window_size,
            start_col,
            price_col
        )
    )

    # =====================================================
    # split by date
    # =====================================================
    train_start, train_end = train_date_ranges['train']

    val_start, val_end = train_date_ranges['val']

    train_mask = (
        (dates_all >= train_start) &
        (dates_all <= train_end)
    )

    val_mask = (
        (dates_all >= val_start) &
        (dates_all <= val_end)
    )

    X_train_i = X_all[train_mask]

    y_train_i = y_all[train_mask]

    X_val_i = X_all[val_mask]

    y_val_i = y_all[val_mask]

    # =====================================================
    # append
    # =====================================================
    if X_train_i.size > 0:

        X_train_list.append(X_train_i)

        y_train_list.append(y_train_i)

    if X_val_i.size > 0:

        X_val_list.append(X_val_i)

        y_val_list.append(y_val_i)

# =========================================================
# 7. test data
# =========================================================

for key in test_indices:

    symbol = indices[key]

    csv_path = f"{csv_folder}/{key}.csv"

    if not os.path.exists(csv_path):

        print("can't find:", csv_path)

        continue

    # =====================================================
    # read csv
    # =====================================================
    df = pd.read_csv(
        csv_path,
        encoding='utf-8-sig'
    )

    if '日期' in df.columns:

        df['日期'] = pd.to_datetime(df['日期'])

        df.set_index('日期', inplace=True)

        df.sort_index(inplace=True)

    # =====================================================
    # create windows
    # =====================================================
    X_all, y_all, dates_all = (
        create_windowed_data_with_labels(
            df,
            window_size,
            start_col,
            price_col
        )
    )

    # =====================================================
    # split by date
    # =====================================================
    test_start, test_end = test_date_ranges['test']

    test_mask = (
        (dates_all >= test_start) &
        (dates_all <= test_end)
    )

    X_test_i = X_all[test_mask]

    y_test_i = y_all[test_mask]

    # =====================================================
    # save
    # =====================================================
    if X_test_i.size > 0:

        X_test_dict[key] = X_test_i

        y_test_dict[key] = y_test_i

# =========================================================
# 8. merge train/val data
# =========================================================

X_train = np.concatenate(
    X_train_list,
    axis=0
)

X_val = np.concatenate(
    X_val_list,
    axis=0
)

y_train = np.concatenate(
    y_train_list
)

y_val = np.concatenate(
    y_val_list
)

# =========================================================
# 9. check train/val data
# =========================================================

print("X_train:", X_train.shape)

print("X_val:", X_val.shape)

print("\nTrain Label Distribution:")

unique, counts = np.unique(y_train, return_counts=True)

for u, c in zip(unique, counts):

    print(
        f"Label {u}: "
        f"{c} "
        f"({c / len(y_train):.4f})"
    )

print("\nVal Label Distribution:")

unique, counts = np.unique(y_val, return_counts=True)

for u, c in zip(unique, counts):

    print(
        f"Label {u}: "
        f"{c} "
        f"({c / len(y_val):.4f})"
    )

# =========================================================
# 10. check test data
# =========================================================

for key in X_test_dict:

    y_test_i = y_test_dict[key]

    print(f"\n{indices[key]} ({key})")

    print("X_test:", X_test_dict[key].shape)

    unique, counts = np.unique(
        y_test_i,
        return_counts=True
    )

    for u, c in zip(unique, counts):

        print(
            f"  Label {u}: "
            f"{c} "
            f"({c / len(y_test_i):.4f})"
        )


############################# Cell 2 #############################
train_tf = X_train[:, :, :-1].reshape(X_train.shape[0], -1)
val_tf   = X_val[:, :, :-1].reshape(X_val.shape[0], -1)

print("train:", train_tf.shape)
print("val:", val_tf.shape)

test_tf_dict = {}

for key in X_test_dict:
    X_test_i = X_test_dict[key]

    test_tf_dict[key] = X_test_i[:, :, :-1].reshape(X_test_i.shape[0], -1)

    print(f"{key} ({indices[key]}):", test_tf_dict[key].shape)

############################# Cell 3 #############################
def build_windows(data, window_size=args.window_size, step=4):

    N, F = data.shape
    windows = []

    for start in range(0, F - window_size + 1, step):

        end = start + window_size

        # output window shape: (N, window_size)
        window = data[:, start:end]

        # =========================
        # normalization base: the 7th feature in the window (the last one)
        # =========================
        base = window[:, -7].reshape(-1, 1)

        # 防止除零
        base[base == 0] = 1e-8

        # =========================
        # normalize the window by the base
        # =========================
        window_norm = window / base

        windows.append(window_norm)

    windows = np.stack(windows, axis=1)

    return windows

train_3d = build_windows(train_tf)
val_3d   = build_windows(val_tf)
print("train_3d:", train_3d.shape)
print("val_3d:", val_3d.shape)

test_3d_dict = {}

for key in test_tf_dict:
    test_3d_dict[key] = build_windows(test_tf_dict[key])
    
    print(f"{key} ({indices[key]}):", test_3d_dict[key].shape)

############################# Cell 4 #############################
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


# =========================================================
# angle encoding
# =========================================================
def convert_to_angle(data_3d, anchor_x, anchor_y):

    N, T, F = data_3d.shape

    x_coords = np.arange(1, T + 1).reshape(1, T, 1)

    dx = anchor_x - x_coords
    dy = anchor_y - data_3d

    angles = np.arctan2(dy, dx)

    return angles

# =========================================================
# Fisher Ratio
# =========================================================
def fisher_ratio(features, labels):

    classes = np.unique(labels)

    global_mean = np.mean(features, axis=0)

    Sb = 0.0
    Sw = 0.0

    for c in classes:
        Xc = features[labels == c]

        class_mean = np.mean(Xc, axis=0)

        Sb += len(Xc) * np.sum((class_mean - global_mean) ** 2)
        Sw += np.sum((Xc - class_mean) ** 2)

    return Sb / (Sw + 1e-12)


# =========================================================
# search Best Anchor
# =========================================================
def search_best_anchor(
    data_3d,
    labels,
    pca_dim=64,
    x_range=(1, 30),
    y_steps=30,
):

    N, T, F = data_3d.shape

    # y range is determined by the min and max of the data
    y_min = data_3d.min()
    y_max = data_3d.max()

    anchor_x_candidates = np.arange(x_range[0], x_range[1])
    anchor_y_candidates = np.linspace(y_min, y_max, y_steps)

    best_score = -np.inf
    best_anchor = None

    results = []

    print("\n==============================")
    print("SEARCHING BEST ANCHOR")
    print("==============================")

    for ax in tqdm(anchor_x_candidates):

        for ay in anchor_y_candidates:

            # 1️⃣ angle transform
            angle = convert_to_angle(data_3d, ax, ay)

            # 2️⃣ reshape
            X = angle.reshape(len(angle), -1)

            # 3️⃣ PCA（稳定 + 降噪）
            X = PCA(n_components=pca_dim).fit_transform(X)

            # 4️⃣ Fisher
            score = fisher_ratio(X, labels)

            results.append((ax, ay, score))

            if score > best_score:
                best_score = score
                best_anchor = (ax, ay)

    # sort results by score in descending order
    results = sorted(results, key=lambda x: x[2], reverse=True)

    print("\n==============================")
    print("BEST ANCHOR FOUND")
    print("==============================")
    print("Best anchor:", best_anchor)
    print("Best Fisher:", best_score)

    return best_anchor, best_score, results


# =========================================================
# import and run the search
# =========================================================

best_anchor, best_score, results = search_best_anchor(
    train_3d,
    y_train,
    pca_dim=64,
    x_range=(train_3d.shape[1]+1, train_3d.shape[1] + 2),
    y_steps=30
)


# =========================================================
# print top 10 anchors for analysis
# =========================================================
print("\nTop 10 anchors:")
for i in range(10):
    ax, ay, score = results[i]
    print(f"{i+1}: ax={ax:.2f}, ay={ay:.4f}, score={score:.6f}")


############################# Cell 5 #############################
import numpy as np

# =========================================================
# angle encoding with best anchor
# =========================================================
def convert_to_angle(data_3d, anchor_x, anchor_y, degree=False):

    N, T, F = data_3d.shape

    x_coords = np.arange(1, T + 1).reshape(1, T, 1)

    dx = anchor_x - x_coords
    dy = anchor_y - data_3d

    angles = np.arctan2(dy, dx)

    if degree:
        angles = np.degrees(angles)

    return angles


# =========================================================
# use the best anchor to convert all datasets
# =========================================================
best_anchor_x = best_anchor[0]
best_anchor_y = best_anchor[1]


# =========================
# TRAIN
# =========================
train_angle = convert_to_angle(
    train_3d,
    best_anchor_x,
    best_anchor_y
)

# =========================
# VAL
# =========================
val_angle = convert_to_angle(
    val_3d,
    best_anchor_x,
    best_anchor_y
)

print("train_angle:", train_angle.shape)
print("val_angle:", val_angle.shape)


# =========================
# TEST
# =========================
test_angle_dict = {}

for key in test_3d_dict:

    test_angle_dict[key] = convert_to_angle(
        test_3d_dict[key],
        best_anchor_x,
        best_anchor_y
    )

    print(f"{key} ({indices[key]}):", test_angle_dict[key].shape)


############################# Cell 6 #############################
from sklearn.cluster import KMeans
import numpy as np

# =========================================================
# 1️⃣ threshold
# =========================================================

low_raw = LOW_THRESHOLD
high_raw = HIGH_THRESHOLD

# =========================================================
# 2️⃣ angle threshold
# threshold points:
# (10, threshold)
#
# anchor:
# (11, 1)
# =========================================================

anchor_x = train_angle.shape[1] + 1
anchor_y = best_anchor[1]

threshold_x = train_angle.shape[1]

low_th = np.arctan2(
    anchor_y - low_raw,
    anchor_x - threshold_x
)

high_th = np.arctan2(
    anchor_y - high_raw,
    anchor_x - threshold_x
)

print("low_th angle :", low_th)
print("high_th angle:", high_th)

# =========================================================
# search parameters
# =========================================================
seed_candidates = np.arange(0, 11)
cluster_candidates = np.arange(100, 1001, 50)

base_seed = 42

# =========================================================
# TRAIN DATA
# use the angle features for clustering
# =========================================================

N_tr, T, D = train_angle.shape

train_2d = train_angle.reshape(-1, D)

# =========================================================
# score function
# =========================================================

def compute_score(labels, centers):

    # angle feature
    center_special_feature = centers[:, -3]

    # =====================================================
    # low / high clusters
    # =====================================================

    low_center_ids = np.where(
        center_special_feature >= low_th
    )[0]

    high_center_ids = np.where(
        center_special_feature <= high_th
    )[0]

    # =====================================================
    # sample counts
    # =====================================================

    low_mask = np.isin(labels, low_center_ids)

    high_mask = np.isin(labels, high_center_ids)

    low_sample_count = low_mask.sum()

    high_sample_count = high_mask.sum()

    # =====================================================
    # score
    # =====================================================

    total_samples = (
        low_sample_count +
        high_sample_count
    )

    diff = abs(
        low_sample_count -
        high_sample_count
    )

    score = total_samples / (diff + 1)

    return (
        score,
        low_sample_count,
        high_sample_count,
        total_samples,
        diff,
        low_center_ids,
        high_center_ids
    )

# =========================================================
# Stage 1:
# search best K with base seed
# =========================================================

print("\n==============================")
print("STAGE 1: SEARCH BEST K")
print("==============================")

best_k = None

best_score = -np.inf

stage1_results = []

for K in cluster_candidates:

    print(f"\n========== K = {K} ==========")

    kmeans = KMeans(
        n_clusters=K,
        random_state=base_seed,
        n_init="auto"
    )

    labels = kmeans.fit_predict(train_2d)

    (
        score,
        low_count,
        high_count,
        total,
        diff,
        low_ids,
        high_ids
    ) = compute_score(
        labels,
        kmeans.cluster_centers_
    )

    print(f"low_samples : {low_count}")
    print(f"high_samples: {high_count}")
    print(f"total       : {total}")
    print(f"diff        : {diff}")
    print(f"score       : {score:.4f}")

    stage1_results.append({
        "K": K,
        "score": score,
        "low": low_count,
        "high": high_count,
        "total": total,
        "diff": diff
    })

    if score > best_score:

        best_score = score

        best_k = K

print("\n==============================")
print("BEST K")
print("==============================")

print("best_k    :", best_k)

print("best_score:", best_score)

# =========================================================
# Stage 2:
# search best seed with best K
# =========================================================
print("\n==============================")
print("STAGE 2: SEARCH BEST SEED")
print("==============================")

best_seed = base_seed 
best_seed_score = best_score
best_kmeans = None

stage2_results = []

for seed in seed_candidates:

    print(f"\n========== seed = {seed} ==========")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=seed,
        n_init="auto"
    )

    labels = kmeans.fit_predict(train_2d)

    (
        score,
        low_count,
        high_count,
        total,
        diff,
        low_ids,
        high_ids
    ) = compute_score(
        labels,
        kmeans.cluster_centers_
    )

    print(f"low_samples : {low_count}")
    print(f"high_samples: {high_count}")
    print(f"total       : {total}")
    print(f"diff        : {diff}")
    print(f"score       : {score:.4f}")

    stage2_results.append({
        "seed": seed,
        "score": score,
        "low": low_count,
        "high": high_count,
        "total": total,
        "diff": diff
    })

    if score > best_seed_score:

        best_seed_score = score
        best_seed = seed
        best_kmeans = kmeans

print("\n==============================")
print("FINAL BEST PARAMS")
print("==============================")

print("best_k       :", best_k)
print("best_seed    :", best_seed)
print("best_score   :", best_seed_score)

############################# Cell 7 #############################
from sklearn.cluster import KMeans
import numpy as np

# =========================================================
# 1️⃣ threshold
# =========================================================

low_raw = LOW_THRESHOLD
high_raw = HIGH_THRESHOLD

# =========================================================
# 2️⃣ angle threshold
#
# threshold points:
# (10, threshold)
#
# anchor:
# (11, 1)
# =========================================================

anchor_x = train_angle.shape[1] + 1
anchor_y = best_anchor[1]

threshold_x = train_angle.shape[1]

low_th = np.arctan2(
    anchor_y - low_raw,
    anchor_x - threshold_x
)

high_th = np.arctan2(
    anchor_y - high_raw,
    anchor_x - threshold_x
)

print("low_th angle :", low_th)
print("high_th angle:", high_th)

# =========================================================
# parameters
# =========================================================

n_total_clusters = best_k

# 全局聚类数量
n_kmeans_clusters = n_total_clusters

LOW_CLUSTER_ID = 0
HIGH_CLUSTER_ID = 1

KMEANS_OFFSET = 2

# =========================================================
# KMeans
# =========================================================

kmeans = KMeans(
    n_clusters=n_kmeans_clusters,
    random_state=best_seed,
    n_init="auto"
)

# =========================================================
# TRAIN
# =========================================================

N_tr, T, D = train_angle.shape

# =========================================================
# (N*T, D)
# =========================================================

train_2d = train_angle.reshape(-1, D)

# =========================================================
# 1️⃣ KMeans
# =========================================================

kmeans_labels = kmeans.fit_predict(train_2d)

# =========================================================
# 2️⃣ cluster centers and their special feature
# =========================================================

centers = kmeans.cluster_centers_

# use the 3rd last feature of the center for regime classification
center_special_feature = centers[:, -3]

# =========================================================
# 3️⃣ lookup table
# =========================================================

lookup = np.arange(
    n_kmeans_clusters
) + KMEANS_OFFSET

# =========================================================

low_center_ids = np.where(
    center_special_feature >= low_th
)[0]

high_center_ids = np.where(
    center_special_feature <= high_th
)[0]

# =========================================================
# 4️⃣ assign regime cluster ids
# =========================================================

lookup[low_center_ids] = LOW_CLUSTER_ID

lookup[high_center_ids] = HIGH_CLUSTER_ID

# =========================================================
# 5️⃣ train labels
# =========================================================

train_cluster_flat = lookup[kmeans_labels]

train_cluster = train_cluster_flat.reshape(N_tr, T)

print("train_cluster:", train_cluster.shape)

# =========================================================
# VAL
# =========================================================

N_val = val_angle.shape[0]

val_2d = val_angle.reshape(-1, D)

# original kmeans labels (not used directly)
val_kmeans_labels = kmeans.predict(val_2d)

# lookup mapping to cluster ids
val_cluster_flat = lookup[val_kmeans_labels]

val_cluster = val_cluster_flat.reshape(N_val, T)

print("val_cluster:", val_cluster.shape)

# =========================================================
# TEST
# =========================================================

test_cluster_dict = {}

for key in test_angle_dict:

    data = test_angle_dict[key]

    N_test = data.shape[0]

    test_2d = data.reshape(-1, D)

    # original kmeans labels (not used directly)
    test_kmeans_labels = kmeans.predict(test_2d)

    # lookup mapping to cluster ids
    cluster_flat = lookup[test_kmeans_labels]

    test_cluster_dict[key] = cluster_flat.reshape(
        N_test,
        T
    )

    print(
        f"{key} ({indices[key]}):",
        test_cluster_dict[key].shape
    )

############################# Cell 8 #############################

import torch

# train_cluster and val_cluster are already in shape (N, T)
train_data = torch.tensor(train_cluster.ravel() , dtype=torch.float32)
val_data   = torch.tensor(val_cluster.ravel(), dtype=torch.float32)

print(f"train_data shape: {train_data.shape}")
print(f"val_data shape:   {val_data.shape}")

############################# Cell 9 #############################
# Hyperparameters
batch_size = 4  # How many batches per training step
context_length = train_cluster.shape[1] - 1  # Length of the token chunk each batch
d_model = 64  # The size of our model token embeddings
num_blocks = 4  # Number of transformer blocks
num_heads = 1  # Number of heads in Multi-head attention
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 8000  # Total of training iterations <- Change this to smaller number for testing
eval_interval = 50  # How often to evaluate
eval_iters = 20
max_token_value = n_total_clusters + 2  # Number of iterations to average for evaluation
device = args.device if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = args.torch_seed
torch.manual_seed(TORCH_SEED)

############################# Cell 9 #############################
import os
import requests
import math
import torch.nn as nn
from torch.nn import functional as F
# Define Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)

# Define Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
       
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        self.dropout_layer = nn.Dropout(self.dropout)
        
        
    def forward(self, x):
        B, T, C = x.shape

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)


        logits = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        logits = logits.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = F.softmax(logits, dim=-1)

        weights = self.dropout_layer(weights)
       
        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out

class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # Note: The order of the operations is different from the original Transformer paper
        # The order here is: LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # Residual connection
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # Set up token embedding look-up table
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value, embedding_dim=self.d_model)

        # Run all the transformer blocks
        # Different from original paper, here we add a final layer norm after all the blocks
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        """
        # Set up position embedding look-up table
        # following the same approach as the original Transformer paper (Sine and Cosine functions)
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # The "logits" are the output values of our model before applying softmax
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            # Crop idx to the max size of positional embeddings table
            idx_crop = idx[:, -self.context_length:]

            # Get predictions
            logits, _ = self(idx_crop)  # logits: (B, T, C)

            # Get the last time step logits
            logits_last_timestep = logits[:, -1, :]  # Shape: (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)

            # Select the most probable index for each sample in the batch
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Shape: (B, 1)

            # Append the sampled indices to idx
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (B, T+1)

        return idx

############################# Cell 10 #############################
# Initialize the model
model = TransformerLanguageModel()
model = model.to(device)

# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=np.rint((len(data)/(context_length+1))- 1).astype(np.int64), size=(batch_size,))*(context_length+1)
    x = torch.stack([data[idx:idx + context_length ] for idx in idxs]).to(device)
    y = torch.stack([data[idx +1:idx + context_length+1] for idx in idxs]).to(device)
    return x, y

# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            
            x_batch = x_batch.long()  
            y_batch = y_batch.long()  
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

############################# Cell 11 #############################
import copy
# Use AdamW optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()

# Early stopping parameters
best_val_loss = float('inf')
best_model_state = None
patience = 300  # Number of evaluations to wait before stopping
patience_counter = 0
early_stop = False

for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3),
              'Validation Loss:', round(losses['valid'].item(), 3))

        # Early stopping check
        current_val_loss = losses['valid'].item()
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict()) # Save best model state
            patience_counter = 0  # Reset patience counter
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
        else:
            patience_counter += 1
            #print(f"No improvement in validation loss for {patience_counter}/{patience} evaluations")
            if patience_counter >= patience:
                early_stop = True
                print(f"Early stopping triggered at step {step}")
                break

    if early_stop:
        break

    xb, yb = get_batch('train')
    xb = xb.long()  
    yb = yb.long()  
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Load the best model weights at the end
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model weights based on validation loss")

print("Training completed")

############################# Cell 12 #############################
import numpy as np
import torch

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    classification_report
)


class ClusterTokenEvaluator:

    def __init__(
        self,
        model,
        device,
        batch_size=3000
    ):

        self.model = model
        self.device = device
        self.batch_size = batch_size

    # =====================================================
    # predict next token
    # =====================================================
    def predict_next_token(
        self,
        cluster_sequence
    ):

        self.model.eval()

        all_predicted_tokens = []

        with torch.no_grad():

            for i in tqdm(
                range(
                    0,
                    len(cluster_sequence),
                    self.batch_size
                )
            ):

                batch = cluster_sequence[
                    i:i+self.batch_size,
                    :-1
                ]

                start_ids = torch.tensor(
                    batch,
                    dtype=torch.long,
                    device=self.device
                )

                y = self.model.generate(
                    start_ids,
                    max_new_tokens=1
                )

                predicted_tokens = y[:, -1]

                all_predicted_tokens.extend(
                    predicted_tokens.cpu().numpy()
                )

        return np.array(all_predicted_tokens)

    # =====================================================
    # token -> class
    # =====================================================
    def token_to_class(
        self,
        predicted_tokens_array
    ):

        #  neutral class=2
        y_pred = np.full(
            shape=len(predicted_tokens_array),
            fill_value=2,
            dtype=np.int32
        )

        # token 0 -> class 0
        y_pred[
            predicted_tokens_array == 0
        ] = 0

        # token 1 -> class 1
        y_pred[
            predicted_tokens_array == 1
        ] = 1

        return y_pred

    # =====================================================
    # evaluate
    # =====================================================
    def evaluate(
        self,
        cluster_sequence,
        y_true
    ):

        # =================================================
        # predict token
        # =================================================
        predicted_tokens_array = (
            self.predict_next_token(
                cluster_sequence
            )
        )

        # =================================================
        # token -> class
        # =================================================
        y_pred = self.token_to_class(
            predicted_tokens_array
        )

        y_true = y_true.astype(int)

        # =================================================
        # distribution
        # =================================================
        print("\n===== Label Distribution =====")

        unique, counts = np.unique(
            y_true,
            return_counts=True
        )

        for u, c in zip(unique, counts):

            print(
                f"Label {u}: "
                f"{c} "
                f"({c / len(y_true):.4f})"
            )

        # =================================================
        # metrics
        # =================================================
        acc = accuracy_score(
            y_true,
            y_pred
        )

        pre = precision_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        # =================================================
        # print
        # =================================================
        print("\n===== 3-Class Evaluation =====")

        print(
            "Accuracy :",
            round(acc, 4)
        )

        print(
            "Precision:",
            round(pre, 4)
        )

        print(
            "Recall   :",
            round(recall, 4)
        )

        print(
            "F1 Score :",
            round(f1, 4)
        )

        print(
            "\nConfusion Matrix:\n",
            cm
        )

        print(
            "\nClassification Report:\n"
        )

        print(
            classification_report(
                y_true,
                y_pred,
                digits=4,
                zero_division=0
            )
        )

        # =================================================
        # return
        # =================================================
        return {

            "acc": acc,

            "precision": pre,

            "recall": recall,

            "f1_score": f1,

            "cm": cm,

            "y_true": y_true,

            "y_pred": y_pred,

            "predicted_tokens": predicted_tokens_array
        }


############################# Cell 13 #############################
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    classification_report
)

import numpy as np

# =========================================================
# Evaluator
# =========================================================
evaluator = ClusterTokenEvaluator(
    model=model,
    device=device,
    batch_size=3000
)

# =========================================================
# 1️⃣ VALIDATION
# =========================================================
print("\n================= VALIDATION =================")

results_val = evaluator.evaluate(
    val_cluster,
    y_val
)

# =========================================================
# 2️⃣ TEST
# =========================================================
results_test_dict = {}

all_y_true = []
all_y_pred = []

print("\n================= TEST =================")

for key in test_cluster_dict:

    print(
        f"\n================= "
        f"{indices[key]} ({key}) "
        f"================="
    )

    X_test_i = test_cluster_dict[key]

    y_test_i = y_test_dict[key]

    # =====================================================
    # evaluate
    # =====================================================
    results = evaluator.evaluate(
        X_test_i,
        y_test_i
    )

    results_test_dict[key] = results

    # =====================================================
    # collect
    # =====================================================
    all_y_true.append(
        results["y_true"]
    )

    all_y_pred.append(
        results["y_pred"]
    )

# =========================================================
# 3️⃣ OVERALL METRICS
# =========================================================
all_y_true = np.concatenate(
    all_y_true
)

all_y_pred = np.concatenate(
    all_y_pred
)

# =========================================================
# metrics
# =========================================================
overall_acc = accuracy_score(
    all_y_true,
    all_y_pred
)

overall_precision = precision_score(
    all_y_true,
    all_y_pred,
    average='macro',
    zero_division=0
)

overall_recall = recall_score(
    all_y_true,
    all_y_pred,
    average='macro',
    zero_division=0
)

overall_f1 = f1_score(
    all_y_true,
    all_y_pred,
    average='macro',
    zero_division=0
)

overall_cm = confusion_matrix(
    all_y_true,
    all_y_pred
)

# =========================================================
# PRINT
# =========================================================
print(
    "\n================= 🌍 OVERALL ================="
)

print(
    "Accuracy :",
    round(overall_acc, 4)
)

print(
    "Precision:",
    round(overall_precision, 4)
)

print(
    "Recall   :",
    round(overall_recall, 4)
)

print(
    "F1 Score :",
    round(overall_f1, 4)
)

print(
    "\nConfusion Matrix:\n",
    overall_cm
)

# =========================================================
# classification report
# =========================================================
print(
    "\nClassification Report:\n"
)
temp_data = classification_report(
    all_y_true,
    all_y_pred,
    digits=4,
    zero_division=0
    )
print(
    temp_data
)
import pickle
output_dir = "results"
with open(f"results/ablation_withoutbest_{args.low_threshold}_{args.torch_seed}_{args.date_class}_{args.window_size}.pkl", "wb") as f:
    pickle.dump(temp_data, f)
