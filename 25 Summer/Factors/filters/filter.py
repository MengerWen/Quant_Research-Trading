import numpy as np
import pandas as pd


@staticmethod
def filter_001_1(df):
    '''衡量当前波动率高低的过滤器'''
    log_ratio = np.log(df['close'] / df['close'].shift(1))
    hv = log_ratio.rolling(20).std()
    return hv


@staticmethod
def filter_001_2(df):
    '''ATR过滤器'''
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    return atr

@staticmethod
def filter_001_3_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
    '''凯尔特纳通道：基于ATR的波动通道'''
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # 计算ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=atr_period, adjust=False).mean()
    
    # 计算通道宽度
    channel_width = multiplier * atr
    return (df['close'] - (ema - channel_width)) / (2 * channel_width)

@staticmethod
def filter_002_1(df):
    '''衡量当前成交量高低的过滤器'''
    volume_mean = df['volume'].rolling(20).mean()
    volume_deviation = (df['volume'] - volume_mean) / volume_mean
    return volume_deviation

@staticmethod
def filter_002_2_obv(df):
    '''能量潮指标：累积成交量平衡'''
    obv = (np.sign(df['close'].diff()) * df['volume'])
    obv = obv.cumsum()
    return obv

@staticmethod
def filter_002_3_vwap(df):
    '''成交量加权平均价'''
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

@staticmethod
def filter_003(df):
    '''衡量当前相对位置高低的过滤器'''
    up = df['high'].rolling(20).max()
    down = df['low'].rolling(20).min()
    price_position = (df['close'] - down) / (up - down)
    return price_position

@staticmethod
def filter_004(df):
    '''衡量短期价格波动快慢的过滤器'''
    std_5 = df['close'].rolling(5).std()
    std_30 = df['close'].rolling(30).std()
    price_fluctuation = std_5 / std_30
    return price_fluctuation


@staticmethod
def filter_005(df):
    '''
    衡量买卖压力的比例的过滤器
    通过taker_buy_volume和总成交量计算买卖不平衡，反映市场中的净买压或卖压。
    '''
    imbalance = 2 * df['taker_buy_volume'] / df['volume'] - 1
    return imbalance

@staticmethod
def filter_006(df):
    '''
    衡量平均交易量的过滤器
    计算每笔交易的平均成交量，反映市场交易规模的大小，指示机构或散户活动。
    '''
    average_trade_size = df['volume'] / df['trade_count']
    return average_trade_size

@staticmethod
def filter_007(df):
    '''
    衡量 candle body 相对价格范围的大小的过滤器
    衡量K线实体的相对大小，反映价格走势的强弱（如强趋势或市场犹豫）。
    '''
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    body_to_range_ratio = body / range_
    return body_to_range_ratio

@staticmethod
def filter_008(df):
    '''衡量上影线相对价格范围的大小的过滤器'''
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    range_ = df['high'] - df['low']
    upper_wick_ratio = upper_wick / range_
    return upper_wick_ratio

@staticmethod
def filter_009(df):
    '''衡量下影线相对价格范围的大小的过滤器'''
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    range_ = df['high'] - df['low']
    lower_wick_ratio = lower_wick / range_
    return lower_wick_ratio

@staticmethod
def filter_010_1(df, period=14):
    '''
    衡量RSI的过滤器，衡量价格的超买或超卖状态。
    '''
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace([np.inf, -np.inf], np.nan).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

@staticmethod
def filter_010_2_mfi(df, period=14):
    '''资金流量指数：结合价格和成交量的RSI变体'''
    # 计算典型价格
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # 计算原始资金流
    raw_money_flow = typical_price * df['volume']
    
    # 计算正向和负向资金流
    money_flow_direction = np.where(typical_price > typical_price.shift(1), 1, -1)
    positive_flow = raw_money_flow.where(money_flow_direction > 0, 0)
    negative_flow = raw_money_flow.where(money_flow_direction < 0, 0)
    
    # 计算资金比率
    money_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
    money_ratio = money_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # 计算MFI
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

@staticmethod
def filter_011(df, short_period=12, long_period=26):
    '''
    衡量MACD的过滤器，可用于识别趋势反转点
    '''
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    return macd

@staticmethod
def filter_012_aroon_up(df, period=14):
    '''阿隆上升指标：衡量价格创新高的能力'''
    # 计算最高价在周期内的位置
    high_idx = df['high'].rolling(period).apply(lambda x: x.argmax(), raw=True)
    aroon_up = 100 * (period - high_idx) / period
    return aroon_up

@staticmethod
def filter_013_aroon_down(df, period=14):
    '''阿隆下降指标：衡量价格创新低的能力'''
    # 计算最低价在周期内的位置
    low_idx = df['low'].rolling(period).apply(lambda x: x.argmin(), raw=True)
    aroon_down = 100 * (period - low_idx) / period
    return aroon_down

@staticmethod
def filter_014_aroon_oscillator(df, period=14):
    '''阿隆震荡器：衡量趋势强度'''
    aroon_up = filter_012_aroon_up(df, period)
    aroon_down = filter_013_aroon_down(df, period)
    return aroon_up - aroon_down

@staticmethod
def filter_015_chaikin_money_flow(df, period=20):
    '''
    Chaikin资金流(CMF)：结合价格和成交量的资金流向指标
    表示资金流入和流出市场的强度，通常用于识别趋势的持续性或反转。
    CMF值大于0表示资金流入，值小于0表示资金流出。
    '''
    # 计算资金流乘数
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 计算资金流量
    money_flow_volume = money_flow_multiplier * df['volume']
    
    # 计算CMF
    cmf = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
    return cmf

@staticmethod
def filter_020_volume_price_trend(df):
    '''量价趋势指标：结合价格变动和成交量'''
    price_change = df['close'].pct_change()
    vpt = (price_change * df['volume']).cumsum()
    return vpt

