import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Short-Term Narrative Scanner", layout="wide")

st.title("⚡ Hyper-Tactical Narrative Scanner (Short-Term)")
st.write("Optimized for 1 to 5-day hold strategies using a 10-day trend + Daily Confirmation filter.")

# Initialize VADER
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
            return 0.0, "ERROR", "Missing API Secret Configuration", 0
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        clean_symbol = ticker_symbol.split('.')[0].strip().upper()
        
        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={thirty_days_ago}&to={today}&token={api_key}"
        response = requests.get(url)
        news_list = response.json()
        
        if not news_list or not isinstance(news_list, list):
            return 0.0, "NEUTRAL", "No news found in last 30 days.", 0
        
        scores = []
        max_headlines = min(len(news_list), 15)
        
        for item in news_list[:max_headlines]:
            headline = item.get('headline', '')
            if headline:
                vs = analyzer.polarity_scores(headline)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", "No text data extracted.", 0
            
        avg_score = sum(scores) / len(scores)
        return round(avg_score, 2), "POSITIVE" if avg_score >= 0.05 else ("NEGATIVE" if avg_score <= -0.05 else "NEUTRAL"), news_list[0].get('headline', 'N/A'), len(scores)
        
    except Exception as e:
        return 0.0, "ERROR", f"Connection Fail: {str(e)}", 0

# ---------------------------------------------------
# FETCH SHORT-TERM TREND & DAILY CONFIRMATION
# ---------------------------------------------------
def get_short_term_metrics(ticker_symbol):
    try:
        # Download slightly more than 10 days to handle weekends/holidays securely
        df = yf.download(ticker_symbol.strip(), period="1mo", interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        if len(close) < 11:
            return None
            
        # 1. 10-Day Trend Calculation
        today_close = float(close.iloc[-1])
        ten_days_ago_close = float(close.iloc[-10])
        trend_10d = round(((today_close - ten_days_ago_close) / ten_days_ago_close) * 100, 2)
        
        # 2. Secondary Filter: Up Today Check (Current close vs Yesterday's close)
        yesterday_close = float(close.iloc[-2])
        is_up_today = today_close > yesterday_close
        daily_change = round(((today_close - yesterday_close) / yesterday_close) * 100, 2)
        
        return trend_10d, is_up_today, daily_change
    except Exception:
        return None

# ------------------------------------
# DYNAMIC INTERFACE CONTROLS
# ------------------------------------
st.sidebar.header("🛠️ Dashboard Controls")
user_input = st.sidebar.text_input("Enter Ticker Symbols (Comma Separated):", value="IWM, AAPL, NVDA, TSLA, MSFT")
watch_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
run_scan = st.sidebar.button("🚀 Run Short-Term Tactical Scan")

# ------------------------------------
# PROCESS SCREENER EXECUTION
# ------------------------------------
if run_scan:
    results = []
    
    with st.spinner("Processing tactical micro-trends..."):
        for ticker in watch_list:
            price_metrics = get_short_term_metrics(ticker)
            if price_metrics is None:
                st.warning(f"⚠️ Insufficient trading history for: **{ticker}**")
                continue
                
            trend_10d, is_up_today, daily_change = price_metrics
            sentiment_score, sentiment_label, top_news, volume = get_aggregated_sentiment(ticker)
            
            # --- HYPER TACTICAL FILTER SELECTION LOGIC ---
            if sentiment_score >= 0.05 and trend_10d > 0:
                if is_up_today:
                    action = "🟢 GREEN LIGHT: Strong Entry (Trend + News + Daily Confirmation)"
                else:
                    action = "🟡 PULLBACK: Wait (Good Trend/News, but Down Today)"
            elif sentiment_score >= 0.05 and trend_10d <= 0:
                if is_up_today:
                    action = "🔵 REVERSAL WATCH: Early Entry Speculation"
                else:
                    action = "📉 VALUE TRAP: Avoid (Good News, Crashing Price)"
            elif sentiment_score < -0.05 and trend_10d > 0:
                action = "⚠️ EXHAUSTION: Danger (Price high, but news turning toxic)"
            else:
                action = "🔴 RED LIGHT: No Setup (Negative/Neutral Momentum)"
                
            results.append({
                "Ticker": ticker,
                "10d Trend (%)": trend_10d,
                "Daily Change (%)": daily_change,
                "Confirmed Up Today?": "✅ Yes" if is_up_today else "❌ No",
                "Sentiment Score": sentiment_score,
                "Tactical Strategy": action,
                "Latest Headline Preview": top_news
            })
            
    if results:
        df_results = pd.DataFrame(results)
        st.write("### 📌 Short-Term Tactical Matrix")
        st.dataframe(df_results, use_container_width=True)
        
        # Matrix Chart Map
        st.write("### 🗺️ Micro-Momentum Map")
        fig = px.scatter(
            df_results, 
            x="Sentiment Score", 
            y="10d Trend (%)",
            text="Ticker",
            color="Tactical Strategy",
            range_x=[-1, 1],
            title="Short-Term Execution Mapping",
            labels={"Sentiment Score": "Media Sentiment Baseline", "10d Trend (%)": "10-Day Momentum Window (%)"}
        )
        fig.update_traces(textposition='top center', marker=dict(size=14))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
