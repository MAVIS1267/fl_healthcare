import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.2):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

 
def initialize_weights(m):
    """Initializes model weights using Kaiming normal initialization."""
    torch.manual_seed(42)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
class DengueDataset(Dataset):
    def __init__(self, data):
        """
            Only take Site, Age, WBC, PLT based on paper explaination
        """
        self.features = torch.tensor(data[['Site', 'Age', 'WBC', 'PLT']].values, dtype=torch.float32)
        self.labels = torch.tensor(data['Dengue'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    data_dir = "../dengue-generalizability/data"
    df = load_all_processed_data(data_dir)
    
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5"]
    if partition_id < len(datasets):
        target_dataset = datasets[partition_id]
        partition_df = df[df['Dataset'].str.contains(target_dataset)].copy()
    else:
        partition_df = df.sample(frac=1.0)
    
    partition_df = partition_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(partition_df, test_size=0.2, random_state=42)
    
    trainloader = DataLoader(DengueDataset(train_df), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(DengueDataset(test_df), batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load entire valid dataset as the centralized evaluation set."""
    data_dir = "/home/fuduweiii/IT/Project/dengue-generalizability/data"
    df = load_all_processed_data(data_dir)
    return DataLoader(DengueDataset(df), batch_size=128)

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def process_set_01(file_path):
    df = pd.read_excel(file_path, engine='xlrd')
    df = df[['SiteNo', 'Age', 'WBC', 'PLT', 'Lab_Confirmed_Dengue']].copy()
    df.columns = ['Site', 'Age', 'WBC', 'PLT', 'Dengue_Raw']
    
    # Author logic: mutate(Dengue = 2 - Dengue) -> 1 is dengue, 2 is no dengue
    df['Dengue'] = 2 - df['Dengue_Raw']
    df['Dataset'] = "Dataset 1"
    return df.drop(columns=['Dengue_Raw']).dropna()

def process_set_02(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df[['Site', 'Age (Converted from Days to Years)', 'LBLEUKRS', 'LBTHRRS', 'FINAL Category']].copy()
    df.columns = ['Site', 'Age', 'WBC', 'PLT', 'Dengue_Cat']
    # Author logic: ifelse(Dengue == "Dengue", 1, 0)
    df['Dengue'] = (df['Dengue_Cat'] == "Dengue").astype(int)
    df['Dataset'] = "Dataset 2"
    return df.drop(columns=['Dengue_Cat']).dropna()

def process_set_03(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df[['age', 'wbc', 'plt', 'dengue']].copy()
    df.columns = ['Age', 'WBC', 'PLT', 'Dengue']
    df['Site'] = 1
    df['Dataset'] = "Dataset 3"
    return df.dropna()

def process_set_04(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    # Filter columns
    df = df[['age2', 'wbc_m3', 'wbc_m1', 'platelets_m3', 'platelets_m1', 'dengue']].copy()
    
    # Author logic: pivot_longer to handle multiple days
    # We'll create two dataframes (Day 3 and Day 1) and combine them
    d3 = df[['age2', 'wbc_m3', 'platelets_m3', 'dengue']].copy()
    d3.columns = ['age_range', 'WBC_raw', 'PLT_raw', 'Dengue']
    d3['Dataset'] = "Dataset 4, Day -3"
    
    d1 = df[['age2', 'wbc_m1', 'platelets_m1', 'dengue']].copy()
    d1.columns = ['age_range', 'WBC_raw', 'PLT_raw', 'Dengue']
    d1['Dataset'] = "Dataset 4, Day -1"
    
    combined = pd.concat([d3, d1], ignore_index=True)
    
    # Author logic: Mean of age range "12-13" -> 12.5
    def parse_age(range_str):
        try:
            parts = re.findall(r'\d+', str(range_str))
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
            return float(parts[0])
        except:
            return None

    combined['Age'] = combined['age_range'].apply(parse_age)
    
    # Author logic: WBC = WBC/1000, PLT = PLT/1000
    combined['WBC'] = combined['WBC_raw'] / 1000
    combined['PLT'] = combined['PLT_raw'] / 1000
    combined['Site'] = 1
    
    return combined[['Site', 'Age', 'WBC', 'PLT', 'Dengue', 'Dataset']].dropna()

def process_set_05(file_path):
    df = pd.read_excel(file_path, engine='xlrd')
    df = df[['AgeEnrol', 'Result_WBC', 'Result_platelet', 'ConfirmDengue_YesNo_FourCriteria']].copy()
    df.columns = ['Age', 'WBC', 'PLT', 'Dengue_Cat']
    # Author logic: ifelse(Dengue == "No dengue", 0, 1)
    df['Dengue'] = (df['Dengue_Cat'] != "No dengue").astype(int)
    df['Site'] = 1
    df['Dataset'] = "Dataset 5"
    return df.drop(columns=['Dengue_Cat']).dropna()

def load_all_processed_data(data_dir):
    """
    Loads and processes all 5 datasets from the specified directory.
    Returns a single concatenated DataFrame.
    """
    loaders = {
        "dengue-data-01.xls": process_set_01,
        "dengue-data-02.xlsx": process_set_02,
        "dengue-data-03.xlsx": process_set_03,
        "dengue-data-04.xlsx": process_set_04,
        "dengue-data-05.xls": process_set_05
    }
    
    all_dfs = []
    for filename, loader_func in loaders.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            print(f"Loading {filename}...")
            all_dfs.append(loader_func(path))
        else:
            print(f"Warning: {filename} not found in {data_dir}")
                
    return pd.concat(all_dfs, ignore_index=True)
