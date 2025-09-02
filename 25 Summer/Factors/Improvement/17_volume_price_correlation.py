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
evaluator=FactorEvaluation(df=df,future_return_periods=10)

# %%
def volume_price_correlation(df, n=20):
    """
    计算成交量变化与价格变动的滚动相关性因子
    1. 计算对数成交量的变化率
    2. 计算价格变动率 (收盘价相对开盘价的变化)
    3. 对两者进行滚动排名
    4. 计算滚动窗口内的相关系数并取负值
    
    参数:
    df: 包含OHLCV数据的DataFrame
    n: 滚动窗口大小 (默认20)
    """
    # 计算log(volume)的一阶差分
    df = df.copy()
    df['log_volume'] = np.log(df['volume'])
    df['delta_log_volume'] = df['log_volume'].diff(1)
    
    # 计算(close-open)/open
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    
    # 创建滚动排名列
    df['rank_delta_log_volume'] = df['delta_log_volume'].rolling(n, min_periods=1).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
    
    df['rank_close_open_ratio'] = df['close_open_ratio'].rolling(n, min_periods=1).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
    
    # 计算滚动相关系数
    factor = df['rank_delta_log_volume'].rolling(n).corr(df['rank_close_open_ratio'])
    
    return factor

# %%
evaluator.set_factor(
    factor_data_or_func=volume_price_correlation,
    factor_name='volume_price_correlation'
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
   IC (Pearson): 0.003351
   Rank_IC (Spearman): -0.038347
📊 信息比率:
   IR: 0.227193
   有效分组数: 10
📊 因子分布:
📋 数据概况:
   数据长度: 130204
   因子列: volume_price_correlation
   收益率列: future_return
   未来收益周期: 10
--------------------------------------------------

🖼️  单币种 (single) 图片展示:
----------------------------------------
📊 显示分组分析图...(分别是Group Return Comparison -Recent vs Historical (20 groups)和volume_price_correlation Distribution Comparison -Recent vs Historical)
'''
