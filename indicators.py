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
    # Ensure 1-D series (handles cases where 'Close' came through as a 1-col DataFrame)
    s = pd.Series(close.squeeze(), index=close.index, dtype="float64")

    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    roll_up = gain.rolling(length, min_periods=length).mean()
    roll_down = loss.rolling(length, min_periods=length).mean()

    # Avoid divide-by-zero
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

def ichimoku(df: pd.DataFrame, conv=9, base=26, span_b=52):
    high = df['High']; low = df['Low']; close = df['Close']
    conv_line = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    base_line = (high.rolling(base).max() + low.rolling(base).min()) / 2
    span_a = ((conv_line + base_line) / 2).shift(base)
    span_b_line = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).shift(base)
    lagging_span = close.shift(-base)
    return conv_line, base_line, span_a, span_b_line, lagging_span
