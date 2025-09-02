# %%
import sys
sys.path.append('/public/src')
from factor_evaluation_server import FactorEvaluation, DataService
import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
from scipy.stats import linregress

# %%
ds=DataService()
df=ds['ETHUSDT_15m_2020_2025']['2021-10-01':]

# %%
# 添加缺失的vwap字段
df['vwap'] = df['turnover'] / df['volume']

# %%
evaluator=FactorEvaluation(df=df,future_return_periods=10)

# %% [markdown]
# # 定义因子！

# %%
class Alpha158Generator:
    def __init__(self, df):
        self.df = df
        self.factor_results = pd.DataFrame(index=df.index)
    
    def _add_factor(self, name, series):
        """安全添加因子到结果集"""
        self.factor_results[name] = series
        
    def generate_all_factors(self):
        """生成所有可实现的Alpha158因子"""
        # 日内因子（13个）
        self.generate_intraday_factors()
        
        # 波动因子（5个窗口 x 1种 = 5个）
        self.generate_volatility_factors()
        
        # 价因子（21种 x 5窗口 = 105个，实际实现85个）
        self.generate_price_factors()
        
        # 量因子（6种 x 5窗口 = 30个，实际实现25个）
        self.generate_volume_factors()
        
        # 量价相关因子（2种 x 5窗口 = 10个）
        self.generate_price_volume_factors()
        
        return self.factor_results
    
    # ============== 因子生成具体实现 ==============
    def generate_intraday_factors(self):
        """日内因子（无窗口参数）"""
        o, h, l, c, vwap = self.df[['open', 'high', 'low', 'close', 'vwap']].values.T
        
        # 计算中间变量
        body = c - o
        shadow_upper = h - np.maximum(o, c)
        shadow_lower = np.minimum(o, c) - l
        total_range = h - l + 1e-12
        
        # 13个日内因子
        self._add_factor('HIGH0', h / c)
        self._add_factor('KLEN', (h - l) / o)
        self._add_factor('KLOW1', shadow_lower / o)
        self._add_factor('KLOW2', shadow_lower / total_range)
        self._add_factor('KMID1', body / o)
        self._add_factor('KMID2', body / total_range)
        self._add_factor('KSFT1', (2*c - h - l) / o)
        self._add_factor('KSFT2', (2*c - h - l) / total_range)
        self._add_factor('KUP1', shadow_upper / o)
        self._add_factor('KUP2', shadow_upper / total_range)
        self._add_factor('LOW0', l / c)
        self._add_factor('OPEN0', o / c)
        self._add_factor('VWAP0', vwap / c)
    
    def generate_volatility_factors(self):
        """波动因子（5个窗口）"""
        windows = [5, 10, 20, 30, 60]
        close = self.df['close']
        
        for w in windows:
            # STD因子
            std = close.rolling(w).std()
            self._add_factor(f'STD_{w}', std / close)
    
    def generate_price_factors(self):
        """价因子（21种类型 x 5窗口）"""
        windows = [5, 10, 20, 30, 60]
        df = self.df
        close = df['close']
        
        for w in windows:
            # 基础滚动计算
            rolling = close.rolling(w)
            high_roll = df['high'].rolling(w)
            low_roll = df['low'].rolling(w)
            
            # 1. BETA (斜率) - 修复：使用自定义滚动回归计算斜率
            def rolling_slope(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                slope, _, _, _, _ = linregress(x, series)
                return slope
                
            beta = close.rolling(w).apply(rolling_slope, raw=False)
            self._add_factor(f'BETA_{w}', beta / close)
            
            # 2. CNTD (涨跌天数差)
            up_days = (close > close.shift(1)).rolling(w).mean()
            down_days = (close < close.shift(1)).rolling(w).mean()
            self._add_factor(f'CNTD_{w}', up_days - down_days)
            
            # 3. CNTP/CNTN (涨/跌天数比例)
            self._add_factor(f'CNTP_{w}', up_days)
            self._add_factor(f'CNTN_{w}', down_days)
            
            # 4. MAX/MIN (极值)
            self._add_factor(f'MAX_{w}', high_roll.max() / close)
            self._add_factor(f'MIN_{w}', low_roll.min() / close)
            
            # 5. ROC (价格变化率)
            self._add_factor(f'ROC_{w}', close.shift(w) / close)
            
            # 6. RSV (随机震荡指标)
            h_max = high_roll.max()
            l_min = low_roll.min()
            self._add_factor(f'RSV_{w}', (close - l_min) / (h_max - l_min + 1e-12))
            
            # 7. SUMP/SUMN (RSI类指标)
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            
            sum_gain = gain.rolling(w).sum()
            sum_loss = loss.rolling(w).sum()
            total = sum_gain + sum_loss + 1e-12
            
            self._add_factor(f'SUMP_{w}', sum_gain / total)
            self._add_factor(f'SUMN_{w}', sum_loss / total)
            
            # 8. SUMD (多空强度差)
            self._add_factor(f'SUMD_{w}', (sum_gain - sum_loss) / total)
            
            # 9. MA (移动平均)
            self._add_factor(f'MA_{w}', ta.sma(close, length=w) / close)
            
            # 10. RANK (价格百分位)
            self._add_factor(f'RANK_{w}', close.rolling(w).rank(pct=True))
            
            # 11. RESI (回归残差)
            def rolling_residuals(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                slope, intercept, _, _, _ = linregress(x, series)
                y_pred = slope * x + intercept
                return series[-1] - y_pred[-1]
                
            resi = close.rolling(w).apply(rolling_residuals, raw=False)
            self._add_factor(f'RESI_{w}', resi / close)
            
            # 12. RSQR (R平方)
            def rolling_rsquare(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                slope, intercept, r_value, _, _ = linregress(x, series)
                return r_value**2
                
            rsqr = close.rolling(w).apply(rolling_rsquare, raw=False)
            self._add_factor(f'RSQR_{w}', rsqr)
            
            # 13. IMAX (最高价出现时间)
            def idx_max(series):
                return np.argmax(series) / len(series) if len(series) > 0 else 0
                
            self._add_factor(f'IMAX_{w}', high_roll.apply(idx_max, raw=True))
            
            # 14. IMIN (最低价出现时间)
            def idx_min(series):
                return np.argmin(series) / len(series) if len(series) > 0 else 0
                
            self._add_factor(f'IMIN_{w}', low_roll.apply(idx_min, raw=True))
            
            # 15. IMXD (高低点时间差) - 修复实现
            # 使用正确的滚动窗口获取方式
            def imxd_calculator(high_series, low_series):
                if len(high_series) == 0 or len(low_series) == 0:
                    return 0
                high_idx = np.argmax(high_series)
                low_idx = np.argmin(low_series)
                return (high_idx - low_idx) / len(high_series)
                
            # 创建空Series存储结果
            imxd = pd.Series(np.nan, index=df.index)
            
            # 遍历每个时间点计算IMXD
            for i in range(w-1, len(df)):
                high_window = df['high'].iloc[i-w+1:i+1].values
                low_window = df['low'].iloc[i-w+1:i+1].values
                imxd.iloc[i] = imxd_calculator(high_window, low_window)
                
            self._add_factor(f'IMXD_{w}', imxd)
            
            # 16. QTLD/QTLU (分位数)
            self._add_factor(f'QTLD_{w}', close.rolling(w).quantile(0.2) / close)
            self._add_factor(f'QTLU_{w}', close.rolling(w).quantile(0.8) / close)
    
    def generate_volume_factors(self):
        """量因子（6种类型 x 5窗口）"""
        windows = [5, 10, 20, 30, 60]
        vol = self.df['volume']
        
        for w in windows:
            # 1. VMA (成交量均线)
            vma = vol.rolling(w).mean()
            self._add_factor(f'VMA_{w}', vma / (vol + 1e-12))
            
            # 2. VSTD (成交量波动率)
            vstd = vol.rolling(w).std()
            self._add_factor(f'VSTD_{w}', vstd / (vol + 1e-12))
            
            # 3. VSUMP/VSUMN (成交量RSI)
            delta = vol.diff()
            v_gain = delta.where(delta > 0, 0)
            v_loss = (-delta).where(delta < 0, 0)
            
            sum_vgain = v_gain.rolling(w).sum()
            sum_vloss = v_loss.rolling(w).sum()
            v_total = sum_vgain + sum_vloss + 1e-12
            
            self._add_factor(f'VSUMP_{w}', sum_vgain / v_total)
            self._add_factor(f'VSUMN_{w}', sum_vloss / v_total)
            
            # 4. VSUMD (成交量多空强度差)
            self._add_factor(f'VSUMD_{w}', (sum_vgain - sum_vloss) / v_total)
            
            # 5. WVMA (成交量加权波动率)
            def weighted_volatility(close_vol):
                if len(close_vol) < 2:
                    return np.nan
                closes = close_vol[:, 0]
                vols = close_vol[:, 1]
                rets = np.abs(np.diff(closes) / closes[:-1])
                weighted_rets = rets * vols[1:]
                return np.std(weighted_rets) / (np.mean(weighted_rets) + 1e-12)
            
            # 使用滚动窗口计算
            wvma = df[['close', 'volume']].rolling(w).apply(
                weighted_volatility, raw=True
            )
            self._add_factor(f'WVMA_{w}', wvma)
    
    def generate_price_volume_factors(self):
        """量价相关因子（2种类型 x 5窗口）"""
        windows = [5, 10, 20, 30, 60]
        df = self.df
        close = df['close']
        vol = df['volume']
        
        for w in windows:
            # 1. CORR (价格-成交量相关性)
            def price_vol_corr(x):
                if len(x) < 2:
                    return np.nan
                prices = x[:, 0]
                volumes = np.log(x[:, 1] + 1)
                return np.corrcoef(prices, volumes)[0, 1]
                
            self._add_factor(
                f'CORR_{w}', 
                df[['close', 'volume']].rolling(w).apply(price_vol_corr, raw=True)
            )
            
            # 2. CORD (价格变化率-成交量变化率相关性)
            def ret_vol_corr(x):
                if len(x) < 2:
                    return np.nan
                prices = x[:, 0]
                volumes = x[:, 1]
                rets = prices[1:] / prices[:-1] - 1
                vol_chg = np.log(volumes[1:] / volumes[:-1] + 1e-12)
                if len(rets) < 1 or len(vol_chg) < 1:
                    return np.nan
                return np.corrcoef(rets, vol_chg)[0, 1]
                
            self._add_factor(
                f'CORD_{w}', 
                df[['close', 'volume']].rolling(w+1).apply(ret_vol_corr, raw=True)
            )

# %%
# 生成所有因子
factor_generator = Alpha158Generator(df)
all_factors = factor_generator.generate_all_factors()

# %% [markdown]
# # 批量执行因子评估

# %%
results = {}

# 遍历所有因子
for factor_name in tqdm(all_factors.columns, desc="Evaluating Factors"):
    try:
        evaluator.set_factor(
            factor_data_or_func=all_factors[factor_name],
            factor_name=factor_name
        )
        results[factor_name] = evaluator.run_full_evaluation(
            run_stationarity_test=False
        )
    except Exception as e:
        print(f"Error evaluating {factor_name}: {str(e)}")
        results[factor_name] = None

# %%
# 结果分析（示例）
successful_factors = [k for k, v in results.items() if v is not None]
print(f"成功评估因子数量: {len(successful_factors)}/{len(all_factors.columns)}")

# %%
# 示例因子分析（可选）
# sig = all_factors['BETA_10']  # 选择一个因子
# future_return = ...  # 获取未来收益
# plt.scatter(sig, future_return)