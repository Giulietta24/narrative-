import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Dynamic Options Radar Pro", layout="wide")

st.title("🛡️ Institutional Confluence & Narrative Options Engine")
st.write("Extracting up to 20 trending tickers dynamically from live market narratives with multi-pillar filtering.")

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ---------------------------------------------------
# STRATEGY EXPLANATION GLOSSARY
# ---------------------------------------------------
with st.expander("📖 Click to view Strategy Key & Pillar Meanings"):
    st.markdown("""
    ### 🧭 The 5 Confluence Pillars Explained
    Your score is calculated out of 5 based on how many of these conditions are met:
    1. **Positive Sentiment:** Aggregate media coverage scores above +0.10.
    2. **10-Day Up-Trend:** The short-term trajectory is moving upward.
    3. **Daily Confirmation:** The price is trading higher than yesterday's close (No falling knives).
    4. **Macro Bull Market:** Price is securely above the 200-day Moving Average (Structural safety).
    5. **Volume Spike (RVOL >= 1.5):** Volume is 50%+ higher than average (Institutional confirmation).

    ### 🧭 Options Target Matching
    * **🟢 BUY LONG CALLS (Score 4-5):** Complete positive confluence. High-volume breakout backed by heavy institutional buying.
    * **🔴 BUY LONG PUTS (Score 0-1 + Negative News):** Complete negative confluence. Perfect structural breakdown environment.
    """)

# ---------------------------------------------------
# DYNAMIC NARRATIVE SCRAPER (Expanded to 20 Tickers)
# ---------------------------------------------------
def get_dynamic_trending_tickers():
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            st.error("Missing Finnhub API Key in Streamlit Secrets.")
            return []
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        response = requests.get(url)
        news_items = response.json()
        
        extracted_tickers = set()
        
        if isinstance(news_items, list):
            for item in news_items:
                text_to_scan = f"{item.get('headline', '')} {item.get('summary', '')}"
                potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', text_to_scan)
                
                # Filter out heavy non-ticker financial acronyms
                blacklist = {"US", "USA", "CEO", "FED", "AI", "SEC", "GDP", "CPI", "IPO", "ETF", "YOY", "UK", "EU"}
                
                for ticker in potential_tickers:
                    if ticker not in blacklist:
                        extracted_tickers.add(ticker)
                        
        # UPGRADED: Expanded limits to process up to 20 unique tickers
        return list(extracted_tickers)[:20]
        
    except Exception as e:
        st.error(f"Error fetching dynamic narrative feed: {e}")
        return []

# ---------------------------------------------------
# SECURE AGGREGATED NEWS SENTIMENT
# ---------------------------------------------------
def get_aggregated_sentiment(ticker_symbol):
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            return 0.0, "ERROR", "Missing Secret API configuration", 0
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        clean_symbol = ticker_symbol.split('.')[0].strip().upper()
        
        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={thirty_days_ago}&to={today}&token={api_key}"
        response = requests.get(url)
        news_list = response.json()
        
        if not news_list or not isinstance(news_list, list):
            return 0.0, "NEUTRAL", "No company news found.", 0
        
        scores = []
        max_headlines = min(len(news_list), 15)
        
        for item in news_list[:max_headlines]:
            headline = item.get('headline', '')
            if headline:
                vs = analyzer.polarity_scores(headline)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", "No text metrics extracted.", 0
            
        avg_score = sum(scores) / len(scores)
        label = "POSITIVE" if avg_score >= 0.05 else ("NEGATIVE" if avg_score <= -0.05 else "NEUTRAL")
        
        # Capture the latest individual breaking headline text
        latest_headline = news_list[0].get('headline', 'N/A')
        return round(avg_score, 2), label, latest_headline, len(scores)
        
    except Exception:
        return 0.0, "ERROR", "Connection Timeout", 0

# ---------------------------------------------------
# CONFLUENCE TECHNICAL CALCULATION UTILITIES
# ---------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_confluence_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol.strip(), period="1y", interval="1d", group_by="column", progress=False)
        if df is None or df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        volume = df["Volume"]
        
        if len(close) < 200:
            return None
            
        today_close = float(close.iloc[-1])
        yesterday_close = float(close.iloc[-2])
        ten_days_ago = float(close.iloc[-10])
        
        trend_10d = round(((today_close - ten_days_ago) / ten_days_ago) * 100, 2)
        is_up_today = today_close > yesterday_close
        
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        above_macro_trend = today_close > sma_200
        
        avg_vol_10d = volume.iloc[-11:-1].mean()
        rvol = round(float(volume.iloc[-1] / (avg_vol_10d + 1e-9)), 2)
        
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

# ---------------------------------------------------
# CONTROLS PANEL SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Target Matrix Controls")
scan_mode = st.sidebar.radio(
    "Select Discovery Feed:",
    ("📰 Live Market Sentiment Feed", "✍️ Manual Custom Entry")
)

if scan_mode == "✍️ Manual Custom Entry":
    user_input = st.sidebar.text_input("Enter Ticker Symbols (Comma Separated):", value="AAPL, NVDA, TSLA, AMD, MSFT, AMZN, NFLX")
    watch_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    with st.spinner("Analyzing real-time global news streams for trending tickers..."):
        watch_list = get_dynamic_trending_tickers()
        
    if watch_list:
        st.sidebar.success(f"Dynamic Watchlist Generated! Found {len(watch_list)} tickers in live news.")
        st.sidebar.write(", ".join(watch_list))
    else:
        watch_list = ["AAPL", "NVDA", "SPY"]
        st.sidebar.warning("Fallback triggered. Using base index anchors.")

run_scan = st.sidebar.button("🛡️ Run Scan Matrix Pipeline")

# ---------------------------------------------------
# APP EXECUTION LOOP
# ---------------------------------------------------
if run_scan:
    results = []
    
    with st.spinner(f"Processing multi-pillar options formulas for {len(watch_list)} discovered assets..."):
        for ticker in watch_list:
            tech = get_confluence_data(ticker)
            if tech is None:
                continue 
                
            sent_score, sent_label, latest_headline, article_vol = get_aggregated_sentiment(ticker)
            
            # --- CONFLUENCE SCORE ACCUMULATION ---
            score_cards = 0
            if sent_score >= 0.10: score_cards += 1
            if tech["trend_10d"] > 0: score_cards += 1
            if tech["is_up_today"]: score_cards += 1
            if tech["above_macro_trend"]: score_cards += 1
            if tech["rvol"] >= 1.5: score_cards += 1
            
            # --- OPTION CLASSIFICATION CONDITIONALS ---
            if score_cards >= 4:
                if tech["rsi"] > 75:
                    action = "🟡 CALL HOLD: Overbought. Wait for minor dip."
                else:
                    action = "🟢 BUY LONG CALLS: High Bullish Confluence aligned."
            elif score_cards <= 1 and sent_score <= -0.10 and not tech["above_macro_trend"]:
                if tech["rsi"] < 25:
                    action = "🟡 PUT HOLD: Oversold. Wait for brief bounce."
                else:
                    action = "🔴 BUY LONG PUTS: High Bearish Confluence aligned."
            elif score_cards == 3:
                action = "🔵 MODERATE MOMENTUM: Missing key confirmation pillars."
            else:
                action = "🚫 NO OPTIONS SETUP: Unreliable chop zone."
                
            results.append({
                "Ticker": ticker,
                "Confluence Score (/5)": score_cards,
                "Options Strategy": action,
                "10d Trend": f"{tech['trend_10d']}%",
                "Volume Spike (RVOL)": tech["rvol"],
                "RSI": tech["rsi"],
                "Macro Bull?": "✅ Yes" if tech["above_macro_trend"] else "❌ No",
                "Sentiment Score": sent_score,
                "Latest Headline Sample": latest_headline  # ADDED: Headline column successfully re-linked
            })
            
    if results:
        df_results = pd.DataFrame(results)
        st.write("### 🎯 Live Options Setup Target Matrix")
        st.dataframe(df_results, use_container_width=True)
        
        # Plotly chart output configurations
        st.write("### 🗺️ Live Options Cluster Distribution Map")
        fig = px.scatter(
            df_results, 
            x="Sentiment Score", 
            y="Volume Spike (RVOL)",
            text="Ticker",
            color="Options Strategy",
            size="RSI",
            range_x=[-1, 1],
            title="Options Distribution Clusters (Bubble size matches RSI)",
            labels={"Sentiment Score": "Media Sentiment", "Volume Spike (RVOL)": "Relative Volume Spikes (RVOL)"}
        )
        
        fig.update_traces(
            textposition='top center', 
            textfont=dict(color='black', size=13, family='Arial Black')
        )
        
        fig.add_hline(y=1.5, line_dash="dash", line_color="black", annotation_text="Institutional Volume Baseline")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid public assets parsed during this specific news window. Hit scan again to cycle the feed.")
