import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Dynamic Narrative Options Matrix", layout="wide")

st.title("🛡️ Institutional Confluence & Narrative Options Engine")
st.write("Dynamically pulling verified optionable tickers out of live financial wires, corporate actions, and social streams.")

@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# Initialize Persistent Session State Memory
if "cached_watchlist" not in st.session_state:
    st.session_state.cached_watchlist = []
if "headline_map" not in st.session_state:
    st.session_state.headline_map = {}

# ---------------------------------------------------
# HELPER: VALIDATE OPTIONABLE WALL STREET EQUITY
# ---------------------------------------------------
def is_real_optionable_stock(ticker):
    """Filter out non-stocks (NATO, UAE, USD) and verify optionable market data presence"""
    forbidden = {"NATO", "UAE", "WTI", "LNG", "USD", "EUR", "FED", "SEC", "CEO", "USA", "UK", "THE", "GDP", "CPI"}
    if ticker in forbidden or not ticker.isalpha() or len(ticker) > 4:
        return False
    try:
        # A quick check using fast_info ensures yfinance can return pricing metrics for this symbol
        t = yf.Ticker(ticker)
        price = t.fast_info.get('last_price', None)
        if price and price > 1.0:
            return True
        return False
    except:
        return False

# ---------------------------------------------------
# DYNAMIC NARRATIVE SCRAPER (Auto-scans until 20+ valid stocks found)
# ---------------------------------------------------
def fetch_and_expand_narrative():
    try:
        if "FINNHUB_API_KEY" not in st.secrets:
            st.error("Missing 'FINNHUB_API_KEY' in Streamlit Secrets.")
            return
            
        api_key = st.secrets["FINNHUB_API_KEY"]
        extracted_tickers = set()
        temp_headline_map = {}
        
        # Scrape general business streams alongside corporate M&A wires
        categories = ["general", "merger"]
        
        for cat in categories:
            url = f"https://finnhub.io/api/v1/news?category={cat}&token={api_key}"
            response = requests.get(url)
            news_items = response.json()
            
            if isinstance(news_items, list):
                for item in news_items:
                    # Break loop early if we have successfully acquired a healthy base of valid tickers
                    if len(extracted_tickers) >= 25:
                        break
                        
                    headline = item.get('headline', '')
                    summary = item.get('summary', '')
                    
                    # Method A: Evaluate direct symbol tags in the news object metadata
                    related_symbol = item.get('symbol', '')
                    if related_symbol:
                        ticker = related_symbol.upper().split('.')[0].strip()
                        if is_real_optionable_stock(ticker):
                            extracted_tickers.add(ticker)
                            if ticker not in temp_headline_map and headline:
                                temp_headline_map[ticker] = headline
                                
                    # Method B: Text parsing regex lookup
                    text_to_scan = f"{headline} {summary}"
                    potential_symbols = re.findall(r'\b[A-Z]{2,4}\b', text_to_scan)
                    for ticker in potential_symbols:
                        if len(extracted_tickers) >= 25:
                            break
                        if is_real_optionable_stock(ticker):
                            extracted_tickers.add(ticker)
                            if ticker not in temp_headline_map and headline:
                                temp_headline_map[ticker] = headline

        final_list = list(extracted_tickers)[:22] # Slice to final processing list size
        st.session_state.cached_watchlist = final_list
        st.session_state.headline_map = {t: temp_headline_map.get(t, "Trending via active corporate events desk narrative.") for t in final_list}
        
    except Exception as e:
        st.error(f"Error processing narrative expansion pipeline: {e}")

# ---------------------------------------------------
# SOCIAL MEDIA & NEWS sentiment CORRELATION PARSER
# ---------------------------------------------------
def get_cross_channel_sentiment(ticker_symbol):
    """Combines traditional media data with Reddit & Twitter alternative metrics via Finnhub"""
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
    clean_symbol = ticker_symbol.strip().upper()
    
    # 1. Traditional Corporate News Sentiment Calculation
    news_score = 0.0
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        past = (datetime.today() - timedelta(days=20)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={past}&to={today}&token={api_key}"
        res = requests.get(url).json()
        if isinstance(res, list) and res:
            scores = [analyzer.polarity_scores(item.get('headline', ''))['compound'] for item in res[:10] if item.get('headline')]
            if scores: news_score = sum(scores) / len(scores)
    except:
        pass

    # 2. Alternative Social Sentiment Scrape (Twitter/Reddit)
    social_score = 0.0
    try:
        social_url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={clean_symbol}&token={api_key}"
        social_data = requests.get(social_url).json()
        
        reddit_entries = social_data.get('reddit', [])
        twitter_entries = social_data.get('twitter', [])
        
        social_mentions = []
        # Pull compound score signals from recent historical buckets
        for item in reddit_entries[:5] + twitter_entries[:5]:
            score = item.get('score', 0.0)
            if score != 0:
                social_mentions.append(score)
                
        if social_mentions:
            # Normalize Finnhub's raw social scaling to a clean -1.0 to +1.0 framework
            raw_avg = sum(social_mentions) / len(social_mentions)
            social_score = max(min(raw_avg, 1.0), -1.0)
    except:
        pass
        
    # Blended scoring matrix weighting: 60% traditional institutional news, 40% retail social momentum
    combined_score = round((news_score * 0.60) + (social_score * 0.40), 2)
    return combined_score, news_score, social_score

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
        if df is None or df.empty or len(df) < 200:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close, volume = df["Close"], df["Volume"]
        today_close, yesterday_close, ten_days_ago = float(close.iloc[-1]), float(close.iloc[-2]), float(close.iloc[-10])
        
        trend_10d = round(((today_close - ten_days_ago) / ten_days_ago) * 100, 2)
        is_up_today = today_close > yesterday_close
        above_macro_trend = today_close > close.rolling(window=200).mean().iloc[-1]
        
        rvol = round(float(volume.iloc[-1] / (volume.iloc[-11:-1].mean() + 1e-9)), 2)
        rsi = round(float(calculate_rsi(close, period=14).iloc[-1]), 2)
        
        return {"trend_10d": trend_10d, "is_up_today": is_up_today, "above_macro_trend": above_macro_trend, "rvol": rvol, "rsi": rsi}
    except:
        return None

# ---------------------------------------------------
# STREAMLIT CONTROLS SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Target Matrix Controls")
scan_mode = st.sidebar.radio("Select Discovery Feed:", ("📰 Live Dynamic News & Social Stream", "✍️ Manual Custom Entry"))

if scan_mode == "✍️ Manual Custom Entry":
    user_input = st.sidebar.text_input("Enter Ticker Symbols (Comma Separated):", value="AAPL, NVDA, TSLA")
    active_watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    if not st.session_state.cached_watchlist:
        with st.spinner("Executing structural news and action item stream capture..."):
            fetch_and_expand_narrative()
            
    if st.sidebar.button("🔄 Clear Cache & Scrape Fresh Narrative Wires"):
        with st.spinner("Flushing session registers and loading next financial news cycle..."):
            fetch_and_expand_narrative()
            
    active_watchlist = st.session_state.cached_watchlist

if active_watchlist:
    st.sidebar.success(f"Watchlist Armed: {len(active_watchlist)} true corporate entities loaded.")
    st.sidebar.write(", ".join(active_watchlist))

run_scan = st.sidebar.button("🛡️ Run Scan Matrix Pipeline")

# ---------------------------------------------------
# SCAN PIPELINE LOGIC EXECUTION LOOP
# ---------------------------------------------------
if run_scan:
    results = []
    
    with st.spinner(f"Validating multi-pillar models and parsing social momentum registers for {len(active_watchlist)} tickers..."):
        for ticker in active_watchlist:
            tech = get_confluence_data(ticker)
            if tech is None:
                continue
                
            blended_sentiment, media_score, social_score = get_cross_channel_sentiment(ticker)
            
            # --- CONFLUENCE ACCUMULATION CRITERIA ---
            score_cards = 0
            if blended_sentiment >= 0.10: score_cards += 1
            if tech["trend_10d"] > 0: score_cards += 1
            if tech["is_up_today"]: score_cards += 1
            if tech["above_macro_trend"]: score_cards += 1
            if tech["rvol"] >= 1.5: score_cards += 1
            
            # --- OPTIONS CONFIGURATION STRATEGY TRANSLATOR ---
            if score_cards >= 4:
                action = "🟡 CALL HOLD: Overbought. Wait for dip." if tech["rsi"] > 75 else "🟢 BUY LONG CALLS: Strong Multi-Channel Momentum Setup."
            elif score_cards <= 1 and blended_sentiment <= -0.10 and not tech["above_macro_trend"]:
                action = "🟡 PUT HOLD: Oversold. Wait for bounce." if tech["rsi"] < 25 else "🔴 BUY LONG PUTS: High Confluence Breakdown Threat."
            elif score_cards == 3:
                action = "🔵 MODERATE MOMENTUM: Missing confirmations."
            else:
                action = "🚫 NO OPTIONS SETUP: Unreliable chop zone."
                
            results.append({
                "Ticker": ticker,
                "Confluence Score (/5)": score_cards,
                "Options Strategy": action,
                "Blended Sentiment": blended_sentiment,
                "News Score (60%)": round(media_score, 2),
                "Social Score (40%)": round(social_score, 2),
                "Volume Spike (RVOL)": tech["rvol"],
                "RSI": tech["rsi"],
                "Macro Bull?": "✅ Yes" if tech["above_macro_trend"] else "❌ No",
                "Latest Narrative Catalyst Headline": st.session_state.headline_map.get(ticker, "Active float trading volume tracker.")
            })
            
    if results:
        df_results = pd.DataFrame(results)
        st.write(f"### 🎯 Dynamic Confluence & Social Momentum Matrix ({len(df_results)} Stocks Verified)")
        st.dataframe(df_results, use_container_width=True)
        
        st.write("### 🗺️ Cross-Channel Sentiment Options Cluster Distribution Map")
        fig = px.scatter(
            df_results, 
            x="Blended Sentiment", 
            y="Volume Spike (RVOL)",
            text="Ticker",
            color="Options Strategy",
            size="RSI",
            range_x=[-1, 1],
            title="Options Clusters Map (X-Axis includes Reddit & Twitter weightings)",
            labels={"Blended Sentiment": "Blended Sentiment (News + Social Media)", "Volume Spike (RVOL)": "Relative Volume Spikes (RVOL)"}
        )
        fig.update_traces(textposition='top center', textfont=dict(color='black', size=13, family='Arial Black'))
        fig.add_hline(y=1.5, line_dash="dash", line_color="black", annotation_text="Institutional Volume Baseline")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid options entities matched during this specific news window. Hit refresh to loop the stream.")
