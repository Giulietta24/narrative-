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
st.write("Extracting up to 20 trending tickers dynamically from live market narratives with persistent headline tracking.")

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# 🛡️ Initialize Session State Memory so things don't disappear on click
if "cached_watchlist" not in st.session_state:
    st.session_state.cached_watchlist = []
if "headline_map" not in st.session_state:
    st.session_state.headline_map = {}

# ---------------------------------------------------
# DYNAMIC NARRATIVE SCRAPER (Saves Headlines in Memory)
# ---------------------------------------------------
def fetch_and_store_narrative():
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            st.error("Missing Finnhub API Key in Streamlit Secrets.")
            return
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        response = requests.get(url)
        news_items = response.json()
        
        extracted_tickers = set()
        temp_headline_map = {}
        
        if isinstance(news_items, list):
            for item in news_items:
                headline = item.get('headline', '')
                summary = item.get('summary', '')
                text_to_scan = f"{headline} {summary}"
                
                # Extract uppercase ticker structures
                potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', text_to_scan)
                blacklist = {"US", "USA", "CEO", "FED", "AI", "SEC", "GDP", "CPI", "IPO", "ETF", "YOY", "UK", "EU"}
                
                for ticker in potential_tickers:
                    if ticker not in blacklist:
                        extracted_tickers.add(ticker)
                        # Save the headline text tied directly to this ticker
                        if ticker not in temp_headline_map and headline:
                            temp_headline_map[ticker] = headline
                            
        # Slice to a maximum of 20 elements
        final_list = list(extracted_tickers)[:20]
        
        # Lock into persistent session state memory
        st.session_state.cached_watchlist = final_list
        st.session_state.headline_map = {t: temp_headline_map.get(t, "Trending in general macro market news stream.") for t in final_list}
        
    except Exception as e:
        st.error(f"Error compiling dynamic narrative stream: {e}")

# ---------------------------------------------------
# AGGREGATED SENTIMENT SCORE CALCULATOR
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
# SIDEBAR PANEL CONTROL INTERFACE
# ---------------------------------------------------
st.sidebar.header("⚙️ Target Matrix Controls")
scan_mode = st.sidebar.radio(
    "Select Discovery Feed:",
    ("📰 Live Market Sentiment Feed", "✍️ Manual Custom Entry")
)

if scan_mode == "✍️ Manual Custom Entry":
    user_input = st.sidebar.text_input("Enter Ticker Symbols (Comma Separated):", value="AAPL, NVDA, TSLA, AMD, MSFT")
    active_watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    # Trigger dynamic generation if memory is completely clean
    if not st.session_state.cached_watchlist:
        with st.spinner("Sucking real-time tickers out of live headlines..."):
            fetch_and_store_narrative()
            
    # Add a refresh button directly in the sidebar to manually cycle the feed
    if st.sidebar.button("🔄 Pull New Financial Headlines"):
        with st.spinner("Scraping fresh news cycles..."):
            fetch_and_store_narrative()
            
    active_watchlist = st.session_state.cached_watchlist
    
    if active_watchlist:
        st.sidebar.success(f"Dynamic Tracking Active: Running {len(active_watchlist)} assets.")
        st.sidebar.write(", ".join(active_watchlist))
    else:
        active_watchlist = ["AAPL", "NVDA", "SPY"]
        st.sidebar.warning("Fallback Mode active. Hit refresh button above.")

run_scan = st.sidebar.button("🛡️ Run Scan Matrix Pipeline")

# ---------------------------------------------------
# RUN SCAN EXECUTION PIPELINE
# ---------------------------------------------------
if run_scan:
    results = []
    
    if not active_watchlist:
        st.warning("Please add tickers or load a live market feed first.")
    else:
        with st.spinner(f"Running full multi-pillar confluence formulas for {len(active_watchlist)} targets..."):
            for ticker in active_watchlist:
                tech = get_confluence_data(ticker)
                if tech is None:
                    continue
                    
                sent_score, sent_label, vol = get_aggregated_sentiment(ticker)
                
                # --- PILLAR COUNTER MATH ---
                score_cards = 0
                if sent_score >= 0.10: score_cards += 1
                if tech["trend_10d"] > 0: score_cards += 1
                if tech["is_up_today"]: score_cards += 1
                if tech["above_macro_trend"]: score_cards += 1
                if tech["rvol"] >= 1.5: score_cards += 1
                
                # --- STRATEGY ASSIGNMENT ---
                if score_cards >= 4:
                    action = "🟡 CALL HOLD: Overbought. Wait for dip." if tech["rsi"] > 75 else "🟢 BUY LONG CALLS: High Bullish Confluence."
                elif score_cards <= 1 and sent_score <= -0.10 and not tech["above_macro_trend"]:
                    action = "🟡 PUT HOLD: Oversold. Wait for bounce." if tech["rsi"] < 25 else "🔴 BUY LONG PUTS: High Bearish Confluence."
                elif score_cards == 3:
                    action = "🔵 MODERATE MOMENTUM: Missing confirmations."
                else:
                    action = "🚫 NO OPTIONS SETUP: Unreliable chop zone."
                
                # Match the ticker back to the original headline captured in memory
                mapped_headline = st.session_state.headline_map.get(ticker, "Extracted from trending financial feed summary.")
                
                results.append({
                    "Ticker": ticker,
                    "Confluence Score (/5)": score_cards,
                    "Options Strategy": action,
                    "10d Trend": f"{tech['trend_10d']}%",
                    "Volume Spike (RVOL)": tech["rvol"],
                    "RSI": tech["rsi"],
                    "Macro Bull?": "✅ Yes" if tech["above_macro_trend"] else "❌ No",
                    "Sentiment Score": sent_score,
                    "Latest Headline Catalyst": mapped_headline  # 🎯 FIXED: Headline will never go missing on scan run!
                })
                
        if results:
            df_results = pd.DataFrame(results)
            st.write("### 🎯 Live Options Setup Target Matrix")
            st.dataframe(df_results, use_container_width=True)
            
            # Scatter Plot Output Configurations
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
            fig.update_traces(textposition='top center', textfont=dict(color='black', size=13, family='Arial Black'))
            fig.add_hline(y=1.5, line_dash="dash", line_color="black", annotation_text="Institutional Volume Baseline")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Historical data lookup failed for the current batch of tickers. Try hitting the refresh button to cycle new stories.")
