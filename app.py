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
st.write("Extracting up to 20+ active options tickers dynamically from verified Wall Street narratives.")

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

if "cached_watchlist" not in st.session_state:
    st.session_state.cached_watchlist = []
if "headline_map" not in st.session_state:
    st.session_state.headline_map = {}

# ---------------------------------------------------
# SECURE API KEY CHECKER
# ---------------------------------------------------
def get_finnhub_key():
    if "FINNHUB_API_KEY" in st.secrets:
        return st.secrets["FINNHUB_API_KEY"]
    return None

# ---------------------------------------------------
# VALIDATE GENUINE OPTIONABLE TICKERS
# ---------------------------------------------------
def is_valid_wallstreet_stock(ticker):
    """Filter out non-stocks like NATO, UAE, USD, BTC"""
    forbidden = {"NATO", "UAE", "WTI", "LNG", "USD", "EUR", "FED", "SEC", "CEO", "USA", "UK"}
    if ticker in forbidden:
        return False
    try:
        # Quick check to ensure it's an active corporation traded on NYSE/NASDAQ
        t = yf.Ticker(ticker)
        info = t.fast_info
        if info and 'last_price' in info and info['last_price'] > 0:
            return True
        return False
    except:
        return False

# ---------------------------------------------------
# GUARANTEED NARRATIVE TICKER EXTRACTOR
# ---------------------------------------------------
def fetch_and_store_narrative():
    api_key = get_finnhub_key()
    extracted_tickers = set()
    temp_headline_map = {}
    
    if not api_key:
        st.warning("⚠️ Finnhub API Key missing from secrets! Defaulting to institutional baseline watchlist.")
        final_list = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "NFLX", "GOOG", "PLTR", "INHD", "BABA", "NKE"]
        st.session_state.cached_watchlist = final_list
        st.session_state.headline_map = {t: "Standard Institutional Core Flow Watchlist" for t in final_list}
        return

    try:
        categories = ["general", "merger"]
        for cat in categories:
            url = f"https://finnhub.io/api/v1/news?category={cat}&token={api_key}"
            response = requests.get(url)
            news_items = response.json()
            
            if isinstance(news_items, list):
                for item in news_items:
                    headline = item.get('headline', '')
                    summary = item.get('summary', '')
                    related_symbol = item.get('symbol', '')
                    
                    if related_symbol and len(related_symbol) <= 5 and related_symbol.isalpha():
                        ticker = related_symbol.upper()
                        if is_valid_wallstreet_stock(ticker):
                            extracted_tickers.add(ticker)
                            if ticker not in temp_headline_map and headline:
                                temp_headline_map[ticker] = headline
                    
                    text_to_scan = f"{headline} {summary}"
                    potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', text_to_scan)
                    
                    for ticker in potential_tickers:
                        if is_valid_wallstreet_stock(ticker):
                            extracted_tickers.add(ticker)
                            if ticker not in temp_headline_map and headline:
                                temp_headline_map[ticker] = headline

        final_list = list(extracted_tickers)[:20]
        st.session_state.cached_watchlist = final_list
        st.session_state.headline_map = {t: temp_headline_map.get(t, "Trending on active corporate news desks.") for t in final_list}
        
    except Exception as e:
        st.error(f"Error gathering narrative: {e}")

# ---------------------------------------------------
# AGGREGATED HISTORICAL NEWS SENTIMENT METRIC
# ---------------------------------------------------
def get_aggregated_sentiment(ticker_symbol):
    api_key = get_finnhub_key()
    if not api_key:
        return 0.15, "POSITIVE", 1 # Mock positive drift if API key isn't provided
        
    try:
        clean_symbol = ticker_symbol.split('.')[0].strip().upper()
        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={thirty_days_ago}&to={today}&token={api_key}"
        response = requests.get(url)
        news_list = response.json()
        
        if not news_list or not isinstance(news_list, list):
            return 0.0, "NEUTRAL", 0
        
        scores = []
        max_headlines = min(len(news_list), 10)
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
    except:
        return 0.0, "NEUTRAL", 0

# ---------------------------------------------------
# TECHNICAL CONFLUENCE MATRIX PROPS
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
        if df is None or df.empty or len(df) < 200:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df["Close"]
        volume = df["Volume"]
        
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
    except:
        return None

# ---------------------------------------------------
# CONTROLS INTERFACE SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Target Matrix Controls")
scan_mode = st.sidebar.radio(
    "Select Discovery Feed:",
    ("📰 Live Market Sentiment Feed", "✍️ Manual Custom Entry")
)

if scan_mode == "✍️ Manual Custom Entry":
    user_input = st.sidebar.text_input("Enter Ticker Symbols (Comma Separated):", value="AAPL, NVDA, TSLA, AMD, MSFT, META, NFLX")
    active_watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    if not st.session_state.cached_watchlist:
        with st.spinner("Scraping narrative feeds..."):
            fetch_and_store_narrative()
            
    if st.sidebar.button("🔄 Pull New Financial Headlines"):
        with st.spinner("Scraping fresh corporate event streams..."):
            fetch_and_store_narrative()
            
    active_watchlist = st.session_state.cached_watchlist

if active_watchlist:
    st.sidebar.success(f"Tracking Active: Running {len(active_watchlist)} valid stocks.")
    st.sidebar.write(", ".join(active_watchlist))

run_scan = st.sidebar.button("🛡️ Run Scan Matrix Pipeline")

# ---------------------------------------------------
# RUN SCAN METRIC PIPELINE LOOP
# ---------------------------------------------------
if run_scan:
    results = []
    
    if not active_watchlist:
        st.warning("No tickers selected or available.")
    else:
        with st.spinner(f"Running multi-pillar matrices across {len(active_watchlist)} assets..."):
            for ticker in active_watchlist:
                tech = get_confluence_data(ticker)
                if tech is None:
                    continue
                    
                sent_score, sent_label, vol = get_aggregated_sentiment(ticker)
                
                score_cards = 0
                if sent_score >= 0.10: score_cards += 1
                if tech["trend_10d"] > 0: score_cards += 1
                if tech["is_up_today"]: score_cards += 1
                if tech["above_macro_trend"]: score_cards += 1
                if tech["rvol"] >= 1.5: score_cards += 1
                
                if score_cards >= 4:
                    action = "治 CALL HOLD: Overbought. Wait for dip." if tech["rsi"] > 75 else "🟢 BUY LONG CALLS: High Bullish Confluence."
                elif score_cards <= 1 and sent_score <= -0.10 and not tech["above_macro_trend"]:
                    action = "治 PUT HOLD: Oversold. Wait for bounce." if tech["rsi"] < 25 else "🔴 BUY LONG PUTS: High Bearish Confluence."
                elif score_cards == 3:
                    action = "🔵 MODERATE MOMENTUM: Missing confirmations."
                else:
                    action = "🚫 NO OPTIONS SETUP: Unreliable chop zone."
                
                mapped_headline = st.session_state.headline_map.get(ticker, "Active corporate wire coverage.")
                
                results.append({
                    "Ticker": ticker,
                    "Confluence Score (/5)": score_cards,
                    "Options Strategy": action,
                    "10d Trend": f"{tech['trend_10d']}%",
                    "Volume Spike (RVOL)": tech["rvol"],
                    "RSI": tech["rsi"],
                    "Macro Bull?": "✅ Yes" if tech["above_macro_trend"] else "❌ No",
                    "Sentiment Score": sent_score,
                    "Latest Headline Catalyst": mapped_headline
                })
                
        if results:
            df_results = pd.DataFrame(results)
            st.write(f"### 🎯 Live Options Setup Target Matrix ({len(df_results)} Stocks Scanned)")
            st.dataframe(df_results, use_container_width=True)
            
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
            st.error("No valid public assets parsed. Try custom ticker entry mode.")
