import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder


NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]


class NSLKDDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_nslkdd(train_path, test_path):
    train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS)

    # Drop difficulty column
    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"], inplace=True)

    # Separate labels
    y_train = train_df["label"]
    y_test = test_df["label"]

    X_train = train_df.drop(columns=["label"])
    X_test = test_df.drop(columns=["label"])

    # One-hot encode categorical features
    X_all = pd.concat([X_train, X_test])
    X_all = pd.get_dummies(X_all)

    X_train = X_all.iloc[:len(X_train)]
    X_test = X_all.iloc[len(X_train):]

    # Encode labels (multi-class)
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train, y_test]))

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, len(label_encoder.classes_)


def get_dataloaders(train_path, test_path, batch_size=64):
    X_train, y_train, X_test, y_test, num_classes = load_nslkdd(
        train_path, test_path
    )

    train_dataset = NSLKDDDataset(X_train, y_train)
    test_dataset = NSLKDDDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape[1], num_classes