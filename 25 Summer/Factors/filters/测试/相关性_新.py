# %%
import sys
sys.path.append('/public/src')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from factor_evaluation_server import FactorEvaluation, DataService
ds = DataService()
df = ds['ETHUSDT_15m_2020_2025']['2021-10-01':]

# %%
path="/public/data/factor_data/ETH_15m_factor_data.txt"
factors=pd.read_csv(path, sep='|')
factors.head()

# %%
for i in list(factors.columns):
    print(i)

# %%
factors.index

# %%
def filter_011(df, short_period=12, long_period=26):
    '''
    衡量MACD的过滤器，可用于识别趋势反转点
    '''
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    return macd

# %%
sig=filter_011(df, 12, 26)

# %%
# sig的index修改为arrange
sig = sig.reset_index(drop=True)

# %%
factors['sig']=sig

# %%
corr_matrix=factors.corr()

# %%
corr_matrix.iloc[-1,:]
