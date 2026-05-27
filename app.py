import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Narrative Sentiment Radar Pro", layout="wide")

st.title("📈 Narrative Sentiment Engine Pro")
st.write("Analyze custom assets using an aggregated sentiment volume matrix paired with 30-day price trends.")

# Initialize VADER
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ---------------------------------------------------
# FETCH AGGREGATED HEADLINES & CALCULATE TOTAL SCORE
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
        # Sample up to 15 headlines for a true aggregate narrative baseline
        max_headlines = min(len(news_list), 15)
        
        for item in news_list[:max_headlines]:
            headline = item.get('headline', '')
            if headline:
                vs = analyzer.polarity_scores(headline)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", "No text data extracted.", 0
            
        # Calculate statistical mean across the headline ecosystem
        avg_score = sum(scores) / len(scores)
        article_count = len(scores)
        
        if avg_score >= 0.05:
            sentiment_label = "POSITIVE"
        elif avg_score <= -0.05:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
            
        # Preview the single latest headline as a reference point
        latest_headline_preview = news_list[0].get('headline', 'N/A')
        return round(avg_score, 2), sentiment_label, latest_headline_preview, article_count
        
    except Exception as e:
        return 0.0, "ERROR", f"Connection Fail: {str(e)}", 0

# ------------------------------------
# FETCH PRICE HISTORY
# ------------------------------------
def get_price_change(ticker_symbol):
    try:
        df = yf.download(ticker_symbol.strip(), period="2mo", interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        if len(close) < 31:
            return None
            
        last = float(close.iloc[-1])
        last_30 = float(close.iloc[-30])
        
        return round(((last - last_30) / last_30) * 100, 2)
    except Exception:
        return None

# ------------------------------------
# DYNAMIC INTERFACE CONTROLS
# ------------------------------------
st.sidebar.header("🛠️ Dashboard Controls")

# Step 1: Input Box for Tickers
user_input = st.sidebar.text_input(
    "Enter Ticker Symbols (Comma Separated):", 
    value="IWM, AAPL, NVDA, TSLA, MSFT"
)

# Convert string input cleanly into a Python list
watch_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]

st.sidebar.write(f"Loaded Tickers: `{watch_list}`")

# Run Button
run_scan = st.sidebar.button("🚀 Run Comprehensive Narrative Scan")

# ------------------------------------
# MAIN PROCESS EXECUTION
# ------------------------------------
if run_scan:
    results = []
    
    with st.spinner(f"Aggregating market headlines and calculating data paths for {len(watch_list)} assets..."):
        for ticker in watch_list:
            price_change = get_price_change(ticker)
            if price_change is None:
                st.warning(f"⚠️ Could not pull price trend history for symbol: **{ticker}**")
                continue
                
            sentiment_score, sentiment_label, top_news, volume = get_aggregated_sentiment(ticker)
            
            # Smart Quadrant Action Formulas using Aggregated Logic
            if sentiment_score >= 0.05 and price_change > 0:
                action = "🚀 Momentum Leader (Buy/Hold)"
            elif sentiment_score >= 0.05 and price_change <= 0:
                action = "🔍 Bullish Divergence (Watch for Dip)"
            elif sentiment_score < -0.05 and price_change > 0:
                action = "⚠️ Bearish Divergence (High Risk Flag)"
            elif sentiment_score <= -0.05 and price_change <= 0:
                action = "📉 Value Trap / Weak (Avoid)"
            else:
                action = "😐 Neutral / Directionless"
                
            results.append({
                "Ticker": ticker,
                "30d Price Change (%)": price_change,
                "Aggregate Sentiment Score": sentiment_score,
                "Headlines Analyzed": volume,
                "Narrative Status": sentiment_label,
                "Tactical Strategy": action,
                "Latest Headline Sample": top_news
            })
            
    if results:
        df_results = pd.DataFrame(results)
        
        # 1. Output Table Block
        st.write("### 📌 Aggregated Narrative Metrics Matrix")
        st.dataframe(df_results, use_container_width=True)
        
        # 2. Plotly Interactive Map Visual
        st.write("### 🗺️ Multi-Asset Narrative Mapping Engine")
        fig = px.scatter(
            df_results, 
            x="Aggregate Sentiment Score", 
            y="30d Price Change (%)",
            text="Ticker",
            color="Tactical Strategy",
            size="Headlines Analyzed", # Visual bubble scaling based on data reliability volume
            range_x=[-1, 1],
            title="Strategic Placement Distribution Map",
            labels={"Aggregate Sentiment Score": "Aggregated Media Sentiment Baseline", "30d Price Change (%)": "30-Day Technical Trajectory (%)"}
        )
        fig.update_traces(textposition='top center')
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid asset entities were parsed during checkout. Check entry nomenclature or API limits.")
