# %%
import sys
sys.path.append('/public/src')
from factor_evaluation_server import FactorEvaluation,DataService # type: ignore
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas_ta as ta

# %%
ds=DataService()
df=ds['ETHUSDT_15m_2020_2025']['2021-10-01':]

# %%
# 添加缺失的vwap字段（如果未提供）
if 'vwap' not in df.columns:
    df['vwap'] = df['turnover'] / df['volume']

# %%
evaluator=FactorEvaluation(df=df,future_return_periods=10)

# %% [markdown]
# # 定义因子！

# %%
# 定义KUP1因子计算函数
def calculate_kup1(df):
    """
    计算KUP1因子（上影线长度相对开盘价的比例）
    公式：KUP1 = (high - max(open, close)) / open
    
    参数:
    df: 包含OHLC数据的DataFrame
    """
    # 计算max(open, close)
    max_open_close = np.maximum(df['open'], df['close'])
    
    # 计算上影线长度
    upper_shadow = df['high'] - max_open_close
    
    # 计算KUP1因子
    kup1 = upper_shadow / df['open']
    
    return kup1

# %% [markdown]
# # 因子测试框架

# %%
evaluator.set_factor(
    factor_data_or_func=calculate_kup1,
    factor_name='calculate_kup1'
)

# %%
result=evaluator.run_full_evaluation(run_stationarity_test=False)
