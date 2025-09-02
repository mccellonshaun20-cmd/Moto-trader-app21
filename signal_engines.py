import numpy as np
import pandas as pd
from indicators import ema, macd, rsi, ichimoku

def technical_signal(df: pd.DataFrame):
    ema5 = ema(df['Close'], 5)
    ema10 = ema(df['Close'], 10)
    ema20 = ema(df['Close'], 20)
    macd_line, macd_sig, macd_hist = macd(df['Close'])
    rsi14 = rsi(df['Close'], 14)
    conv, base, span_a, span_b, lag = ichimoku(df)

    bull = (ema5 > ema10) & (ema10 > ema20) & (macd_line > macd_sig) & (macd_hist > 0) & (rsi14 > 50) & (df['Close'] > span_b.fillna(df['Close']))
    bear = (ema5 < ema10) & (ema10 < ema20) & (macd_line < macd_sig) & (macd_hist < 0) & (rsi14 < 50) & (df['Close'] < span_b.fillna(df['Close']))

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[bull & ~bear] = 1
    sig[bear & ~bull] = -1
    return sig

def combine_signals(tech: int, macro: int, fund: int, w_tech: float, w_macro: float, w_fund: float):
    return float(w_tech * tech + w_macro * macro + w_fund * fund)
