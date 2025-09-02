import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_prices(ticker: str, period: str = '1y', interval: str = '1d'):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.title).dropna()
    return df

def fetch_vix(period='1y', interval='1d'):
    return fetch_prices('^VIX', period, interval)

def fetch_dxy(period='1y', interval='1d'):
    return fetch_prices('DX-Y.NYB', period, interval)
