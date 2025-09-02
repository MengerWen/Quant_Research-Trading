# %%
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
def ma_deviation_factor(df,ma_period=20):
    ma=df['close'].rolling(ma_period).mean()
    return (df['close']-ma)/ma

# %%
evaluator.set_factor(
    factor_data_or_func=ma_deviation_factor,
    factor_name='ma_deviation_factor'
)

# %%
result=evaluator.run_full_evaluation(run_stationarity_test=True)
'''
📊 单币种 (single) 详细评估结果:
--------------------------------------------------
📈 平稳性检验 (ADF):
   p_value: 0.000000
   是否平稳: 是
🔗 相关性分析:
   IC (Pearson): -0.004886
   Rank_IC (Spearman): -0.032447
📊 信息比率:
   IR: -0.318784
   有效分组数: 10
📊 因子分布:
📋 数据概况:
   数据长度: 129287
   因子列: vwap_deviation_factor
   收益率列: future_return
   未来收益周期: 10
--------------------------------------------------

🖼️  单币种 (single) 图片展示:
----------------------------------------
📊 显示分组分析图...(分别是Group Return Comparison -Recent vs Historical (20 groups)和vwap deviation factor Distribution Comparison -Recent vs Historical)
'''