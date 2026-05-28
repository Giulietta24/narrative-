import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="High-Probability Narrative Engine", layout="wide")

st.title("🛡️ Institutional Confluence & Narrative Engine")
st.write("Filtering short-term sentiment alongside Volume Spikes, Macro Trends, and RSI Guardrails.")

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ---------------------------------------------------
# SECURE AGGREGATED NEWS SENTIMENT
# ---------------------------------------------------
def get_aggregated_sentiment(ticker_symbol):
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            return 0.0, "ERROR", 0
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        clean_symbol = ticker_symbol.split('.')[0].strip().upper()
        
        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={thirty_days_ago}&to={today}&token={api_key}"
        response = requests.get(url)
        news_list = response.json()
        
        if not news_list or not isinstance(news_list, list):
            return 0.0, "NEUTRAL", 0
        
        scores = []
        max_headlines = min(len(news_list), 15)
        
        for item in news_list[:max_headlines]:
            headline = item.get('headline', '')
            if headline:
                vs = analyzer.polarity_scores(headline)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", 0
            
        avg_score = sum(scores) / len(scores)
        label = "POSITIVE" if avg_score >= 0.05 else ("NEGATIVE" if avg_score <= -0.05 else "NEUTRAL")
        return round(avg_score, 2), label, len(scores)
        
    except Exception:
        return 0.0, "ERROR", 0

# ---------------------------------------------------
# CONFLUENCE TECH FILTERS (RVOL, SMA, RSI)
# ---------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_confluence_data(ticker_symbol):
    try:
        # Pull 1 year of data to calculate the 200-day Moving Average securely
        df = yf.download(ticker_symbol.strip(), period="1y", interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        volume = df["Volume"]
        
        if len(close) < 200:
            return None
            
        # Core Price Metrics
        today_close = float(close.iloc[-1])
        yesterday_close = float(close.iloc[-2])
        ten_days_ago = float(close.iloc[-10])
        
        trend_10d = round(((today_close - ten_days_ago) / ten_days_ago) * 100, 2)
        is_up_today = today_close > yesterday_close
        
        # Confluence Indicator 1: 200-day Structural Trend Anchor
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        above_macro_trend = today_close > sma_200
        
        # Confluence Indicator 2: Relative Volume (RVOL)
        # Compares today's volume against the 10-day average volume
        avg_vol_10d = volume.iloc[-11:-1].mean()
        rvol = round(float(volume.iloc[-1] / (avg_vol_10d + 1e-9)), 2)
        
        # Confluence Indicator 3: RSI Guardrail
        rsi_series = calculate_rsi(close, period=14)
        current_rsi = round(float(rsi_series.iloc[-1]), 2)
        
        return {
            "trend_10d": trend_10d,
            "is_up_today": is_up_today,
            "above_macro_trend": above_macro_trend,
            "rvol": rvol,
            "rsi": current_rsi
        }
    except Exception:
        return None

# ------------------------------------
# INTERFACE
# ------------------------------------
st.sidebar.header("🛠️ Confluence Panel")
user_input = st.sidebar.text_input("Tickers:", value="AAPL, NVDA, TSLA, MSFT, AMD")
watch_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
run_scan = st.sidebar.button("🛡️ Scan with Confluence Filters")

if run_scan:
    results = []
    
    with st.spinner("Analyzing multi-factor confluence variables..."):
        for ticker in watch_list:
            tech = get_confluence_data(ticker)
            if tech is None:
                st.warning(f"⚠️ Stock skipped (Needs 1-year history for 200-MA mapping): **{ticker}**")
                continue
                
            sent_score, sent_label, vol = get_aggregated_sentiment(ticker)
            
            # --- HIGH PROBABILITY CONFLUENCE SCORING SYSTEM ---
            score_cards = 0
            if sent_score >= 0.10: score_cards += 1  # Solid positive narrative
            if tech["trend_10d"] > 0: score_cards += 1  # 10-day momentum
            if tech["is_up_today"]: score_cards += 1  # Intraday green confirmation
            if tech["above_macro_trend"]: score_cards += 1  # Structural market tide support
            if tech["rvol"] >= 1.5: score_cards += 1  # Institutional footprint present
            
            # Formulate Strict Trading Action Plan
            if score_cards >= 4:
                if tech["rsi"] > 75:
                    action = "🟡 OVERBOUGHT HOLD: High Confluence, but RSI is overextended. Wait for cooling."
                else:
                    action = "🔥 HIGH PROBABILITY BUY: Complete Confluence Aligned."
            elif score_cards == 3:
                action = "🔵 MODERATE MOMENTUM: Decent setup, missing key confirmation pillars."
            else:
                action = "🚫 RED LIGHT: Low Probability. Sub-optimal conditions."
                
            results.append({
                "Ticker": ticker,
                "Confluence Match (/5)": score_cards,
                "Tactical Strategy": action,
                "10d Trend": f"{tech['trend_10d']}%",
                "RVOL (Vol Spike)": tech["rvol"],
                "Current RSI": tech["rsi"],
                "Macro Bull Market?": "✅ Yes" if tech["above_macro_trend"] else "❌ No",
                "Sentiment Score": sent_score
            })
            
    if results:
        df_results = pd.DataFrame(results)
        st.write("### 🛡️ Verified Action Output Matrix")
        st.dataframe(df_results, use_container_width=True)
        
        fig = px.scatter(
            df_results, 
            x="Sentiment Score", 
            y="RVOL (Vol Spike)",
            text="Ticker",
            color="Tactical Strategy",
            size="Current RSI",
            title="Confluence Clustering Map (Bubble size matches RSI value)",
            labels={"Sentiment Score": "Media Sentiment", "RVOL (Vol Spike)": "Relative Volume Multiplier"}
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Institutional Volume Line")
        st.plotly_chart(fig, use_container_width=True)
