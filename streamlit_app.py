import streamlit as st
import pandas as pd
import numpy as np
import json, os, math
import plotly.graph_objects as go
from datetime import datetime

from data_connectors import fetch_prices, fetch_vix, fetch_dxy
from indicators import ema, macd, rsi, ichimoku
from signal_engines import technical_signal, combine_signals
from adapter import AdaptiveWeighter

st.set_page_config(page_title="Moto Trader — Adaptive", layout="wide")

UNIVERSE = ["SPY", "SOFI", "NVDA", "AMD"]

# ----- State & Portfolio -----
PORT_PATH = os.path.join(os.path.dirname(__file__), "portfolio.json")

def load_port():
    try:
        with open(PORT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"cash": 100000.0, "positions": {}, "history": []}

def save_port(p):
    with open(PORT_PATH, "w") as f:
        json.dump(p, f, indent=2)

portfolio = load_port()

# ----- Sidebar Controls -----
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect("Watchlist", UNIVERSE, default=UNIVERSE)
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
st.sidebar.markdown("---")
st.sidebar.subheader("Adaptive Weights")
w_tech = st.sidebar.slider("Technical weight", 0.0, 1.0, 0.5, 0.05)
w_macro = st.sidebar.slider("Macro weight", 0.0, 1.0, 0.3, 0.05)
w_fund  = st.sidebar.slider("Fundamental weight", 0.0, 1.0, 0.2, 0.05)
st.sidebar.markdown("---")
risk_per_trade = st.sidebar.slider("Max risk per trade (weight cap)", 0.0, 0.25, 0.05, 0.01)
auto_trade = st.sidebar.checkbox("AutoTrade: align to target weight on signal", value=False)

# ----- Helper: Plot Candles + EMAs + Ichimoku -----
def plot_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=df.index, y=ema(df["Close"], 5),  mode="lines", name="EMA 5"))
    fig.add_trace(go.Scatter(x=df.index, y=ema(df["Close"], 10), mode="lines", name="EMA 10"))
    fig.add_trace(go.Scatter(x=df.index, y=ema(df["Close"], 20), mode="lines", name="EMA 20"))
    conv, base, span_a, span_b, lag = ichimoku(df)
    fig.add_trace(go.Scatter(x=df.index, y=span_a, mode="lines", name="Span A", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=span_b, mode="lines", name="Span B", line=dict(width=1), fill='tonexty', fillcolor='rgba(0, 150, 0, 0.1)'))
    fig.update_layout(title=title, height=500, xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig

def plot_macd_rsi(df: pd.DataFrame):
    macd_line, macd_sig, macd_hist = macd(df["Close"])
    rsi14 = rsi(df["Close"], 14)
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=df.index, y=macd_hist, name="MACD Hist"))
    f1.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD"))
    f1.add_trace(go.Scatter(x=df.index, y=macd_sig,  name="Signal"))
    f1.update_layout(title="MACD", height=250, legend_orientation="h")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df.index, y=rsi14, name="RSI(14)"))
    f2.add_hline(y=70, line_dash="dash")
    f2.add_hline(y=30, line_dash="dash")
    f2.update_layout(title="RSI", height=250, legend_orientation="h")
    return f1, f2

# ----- Macro Proxies (robust to rate limits) -----
@st.cache_data(show_spinner=False)
def load_macro(period, interval):
    try:
        vix = fetch_vix(period, interval)["Close"].pct_change().fillna(0)
    except Exception:
        vix = pd.Series(dtype="float64")
    try:
        dxy = fetch_dxy(period, interval)["Close"].pct_change().fillna(0)
    except Exception:
        dxy = pd.Series(dtype="float64")
    return vix, dxy

vix_chg, dxy_chg = load_macro(period, interval)

# ----- Main Dashboard -----
st.title("📈 Moto Trader — Adaptive, Macro-Aware System")
col_a, col_b, col_c = st.columns([1, 1, 1])
col_a.metric("Cash", f"${portfolio['cash']:,.2f}")
mv = 0.0
for sym, pos in portfolio["positions"].items():
    try:
        px = fetch_prices(sym, period="1mo", interval="1d")["Close"].iloc[-1]
        mv += pos["qty"] * px
    except Exception:
        pass
col_b.metric("Market Value", f"${mv:,.2f}")
col_c.metric("Equity", f"${portfolio['cash'] + mv:,.2f}")

st.markdown("---")

for t in tickers:
    st.subheader(f"{t}")
    df = fetch_prices(t, period=period, interval=interval)
    if df.empty:
        st.warning("No data loaded.")
        continue

    # Guard: skip if not enough bars for indicators
    if len(df) < 60:
        st.info("Not enough bars yet for signals on this interval.")
        continue

    # Compute signals
    tech_sig_series = technical_signal(df)
    tech_sig = int(tech_sig_series.iloc[-1])

    # Macro directional proxy: inverse VIX + inverse DXY on latest aligned bar
    common_index = df.index.intersection(vix_chg.index)
    vix_latest = vix_chg.reindex(common_index).iloc[-1] if len(common_index) > 0 else 0
    dxy_latest = dxy_chg.reindex(common_index).iloc[-1] if len(common_index) > 0 else 0
    macro_sig = int(np.sign(-vix_latest - 0.5 * dxy_latest)) if (vix_latest != 0 or dxy_latest != 0) else 0

    fund_sig = 0  # placeholder for earnings/news/forward EPS APIs

    combined = combine_signals(tech_sig, macro_sig, fund_sig, w_tech, w_macro, w_fund)
    target_weight = math.tanh(1.5 * combined) * risk_per_trade

    # Current position
    pos = portfolio["positions"].get(t, {"qty": 0, "avg": 0.0})
    last_px = float(df["Close"].iloc[-1])
    pos_val = pos["qty"] * last_px
    equity = portfolio["cash"] + mv
    current_weight = (pos_val / equity) if equity > 0 else 0.0

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(plot_chart(df, f"{t} Price with EMAs & Ichimoku"), use_container_width=True)
        f_macd, f_rsi = plot_macd_rsi(df)
        st.plotly_chart(f_macd, use_container_width=True)
        st.plotly_chart(f_rsi, use_container_width=True)

    with c2:
        st.markdown("#### Signals")
        st.write(f"Technical: **{tech_sig:+d}**, Macro: **{macro_sig:+d}**, Fundamental: **{fund_sig:+d}**")
        st.write(f"Combined score: **{combined:+.2f}** → Target weight: **{target_weight:+.2%}**")
        st.progress(min(1.0, max(0.0, (combined + 1) / 2)))
        st.markdown("#### Position")
        st.write(f"Qty: **{pos['qty']}**, Last: **${last_px:,.2f}**, MV: **${pos_val:,.2f}**")
        st.write(f"Current weight: **{current_weight:+.2%}**")

        # Order ticket
        st.markdown("#### Trade Ticket")
        colx, coly = st.columns(2)
        qty = colx.number_input("Quantity", value=1, min_value=1, step=1)
        if colx.button("Buy", key=f"buy_{t}"):
            cost = qty * last_px
            if portfolio["cash"] >= cost:
                new_qty = pos["qty"] + qty
                new_avg = (pos["avg"] * pos["qty"] + cost) / max(1, new_qty)
                portfolio["cash"] -= cost
                portfolio["positions"][t] = {"qty": new_qty, "avg": new_avg}
                portfolio["history"].append({"time": datetime.now().isoformat(), "sym": t, "side": "BUY", "qty": qty, "px": last_px})
                save_port(portfolio)
                st.success(f"Bought {qty} {t} @ ${last_px:.2f}")
            else:
                st.error("Insufficient cash")
        if coly.button("Sell", key=f"sell_{t}"):
            if pos["qty"] >= qty:
                proceeds = qty * last_px
                portfolio["cash"] += proceeds
                portfolio["positions"][t]["qty"] = pos["qty"] - qty
                portfolio["history"].append({"time": datetime.now().isoformat(), "sym": t, "side": "SELL", "qty": qty, "px": last_px})
                save_port(portfolio)
                st.success(f"Sold {qty} {t} @ ${last_px:.2f}")
            else:
                st.error("Not enough shares")

        # AutoTrade aligner
        if auto_trade:
            desired_val = target_weight * (portfolio["cash"] + mv)
            delta_val = desired_val - pos_val
            target_qty_delta = int(delta_val / last_px)
            if target_qty_delta != 0:
                if target_qty_delta > 0:
                    cost = target_qty_delta * last_px
                    if portfolio["cash"] >= cost:
                        new_qty = pos["qty"] + target_qty_delta
                        new_avg = (pos["avg"] * pos["qty"] + cost) / max(1, new_qty)
                        portfolio["cash"] -= cost
                        portfolio["positions"][t] = {"qty": new_qty, "avg": new_avg}
                        portfolio["history"].append({"time": datetime.now().isoformat(), "sym": t, "side": "BUY", "qty": target_qty_delta, "px": last_px, "reason": "AutoTrade"})
                        save_port(portfolio)
                        st.info(f"AutoTrade: bought {target_qty_delta} {t}")
                else:
                    sell_qty = min(pos["qty"], abs(target_qty_delta))
                    if sell_qty > 0:
                        proceeds = sell_qty * last_px
                        portfolio["cash"] += proceeds
                        portfolio["positions"][t]["qty"] = pos["qty"] - sell_qty
                        portfolio["history"].append({"time": datetime.now().isoformat(), "sym": t, "side": "SELL", "qty": sell_qty, "px": last_px, "reason": "AutoTrade"})
                        save_port(portfolio)
                        st.info(f"AutoTrade: sold {sell_qty} {t}")

st.markdown("---")
st.subheader("📒 Trade History")
hist = pd.DataFrame(portfolio["history"])
if not hist.empty:
    st.dataframe(hist.sort_values("time", ascending=False), use_container_width=True)
else:
    st.info("No trades yet.")

st.caption("Educational demo — not investment advice. Data via Yahoo Finance (yfinance).")
