# %%
import sys
import os

# 添加 factor_evaluation.pyd 所在的目录到系统路径
pyd_path = r"D:\MG\！internship\！4L CAPITAL\因子评估器"
sys.path.append(pyd_path)

# 导入模块
from factor_evaluation import FactorEvaluation
from factor_evaluation import DataService
import numpy as np
import pandas as pd

# %%
ds=DataService()
df=ds['ETHUSDT_15m_2020_2025']['2021-10-01':]

# %%
evaluator=FactorEvaluation(df=df,future_return_periods=10)

# %%
def calculate_atr(df, period=20):
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    df.drop(['high_low', 'high_close_prev', 'low_close_prev', 'TR'], axis=1, inplace=True)
    return df

# %%
evaluator.set_factor(
    factor_data_or_func=calculate_atr,
    factor_name='calculate_atr'
)

'''
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File factor_evaluation.py:590, in factor_evaluation.FactorEvaluation.set_factor()

ValueError: 因子函数必须返回pandas.Series或numpy.array类型

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
Cell In[5], line 1
----> 1 evaluator.set_factor(
      2     factor_data_or_func=calculate_atr,
      3     factor_name='calculate_atr'
      4 )

File factor_evaluation.py:594, in factor_evaluation.FactorEvaluation.set_factor()

ValueError: 因子函数执行失败: 因子函数必须返回pandas.Series或numpy.array类型
'''

# %%
result=evaluator.run_full_evaluation(run_stationarity_test=True)
