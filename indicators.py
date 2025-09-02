import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(close: pd.Series, length=14):
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(length).mean()
    roll_down = pd.Series(down, index=close.index).rolling(length).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ichimoku(df: pd.DataFrame, conv=9, base=26, span_b=52):
    high = df['High']; low = df['Low']; close = df['Close']
    conv_line = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    base_line = (high.rolling(base).max() + low.rolling(base).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(base)
    span_b_line = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).shift(base)
    lagging_span = close.shift(-base)
    return conv_line, base_line, span_a, span_b_line, lagging_span
