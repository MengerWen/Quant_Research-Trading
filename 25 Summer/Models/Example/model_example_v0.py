import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

import os
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesLSTM:
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1,
                 epochs=10, batch_size=32, lr=0.001, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def _prepare_data(self, X, y=None):
        X_tensor = torch.tensor(X.values.reshape(-1, 1, self.input_size), dtype=torch.float32).to(self.device)
        if y is not None:
            y_tensor = torch.tensor(y.values.reshape(-1, self.output_size), dtype=torch.float32).to(self.device)
            return TensorDataset(X_tensor, y_tensor)
        return TensorDataset(X_tensor)

    def fit(self, X_train, y_train):
        train_dataset = self._prepare_data(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()


    def predict(self, X_test):
        test_dataset = self._prepare_data(X_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                outputs = self.model(batch_X[0])
                predictions.append(outputs.cpu().numpy())
        return np.vstack(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = y.values.reshape(-1, self.output_size)

        mse = np.mean((y_true - y_pred)**2)
        return -mse

def train(X,y,model,n_split = 5,normalize=False):
    scores = []
    tscv = TimeSeriesSplit(n_splits=n_split)
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logging.info(f"--- Fold {fold+1} ---")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if normalize:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        logging.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        logging.info(f"训练集索引范围: {X.index[train_index.min()]}-{X.index[train_index.max()]}")
        logging.info(f"测试集索引范围: {X.index[test_index.min()]}-{X.index[test_index.max()]}")

        current_model = copy.deepcopy(model)

        current_model.fit(X_train, y_train)
        predictions = current_model.predict(X_test)
        is_score = current_model.score(X_train, y_train)
        os_score = current_model.score(X_test, y_test)
        scores.append(os_score)

        logging.info(f"训练集评分: {is_score:.4f}, 测试集评分: {os_score:.4f}")
    return -sum(scores)

if __name__=='__main__':
    factor_data_path = '/public/data/factor_data'
    file_name = 'BTCUSDT_15m_2020_2025_factor_data.pkl'

    data = pd.read_pickle(os.path.join(factor_data_path,file_name))

    data['target'] = data['close'].shift(-10)/data['close'] - 1

    begin = '2021-10-01'
    split = '2025-03-01'
    selected_factors = [f'c_chu0{i}' for i in range(37,52)]
    workding_data = data[selected_factors+['target']][begin:split].dropna()

    X_data = workding_data[selected_factors]
    y_data = workding_data['target']

    ridge_model = Ridge(0.0008)
    lgb_model = lgb.LGBMRegressor(random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")


    lstm_model = TimeSeriesLSTM(
        input_size=len(selected_factors),
        hidden_size=64,
        num_layers=2,
        epochs=5,
        batch_size=32,
        lr=0.001,
        device=device
    )

    logging.info("\n--- Training Ridge Model ---")
    ridge_score_sum = train(X_data, y_data, ridge_model, normalize=True)
    logging.info(f"Ridge Model Total  Score: {ridge_score_sum:.4f}")

    logging.info("\n--- Training LightGBM Model ---")
    lgb_score_sum = train(X_data, y_data, lgb_model, normalize=False)
    logging.info(f"LightGBM Model Total  Score: {lgb_score_sum:.4f}")

    logging.info("\n--- Training LSTM Model ---")
    lstm_score_sum = train(X_data, y_data, lstm_model, normalize=True)
    logging.info(f"LSTM Model Total  Score: {lstm_score_sum:.4f}")