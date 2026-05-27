import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Narrative Sentiment Radar", layout="wide")

st.title("📈 Multi-Asset Narrative Sentiment Radar")
st.write("Cross-referencing secure Finnhub headlines with 30-day technical trends.")

# Initialize VADER
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ------------------------------------
# FETCH REAL-TIME NEWS (SECURE API CALL)
# ------------------------------------
def get_ticker_sentiment(ticker_symbol):
    try:
        api_key = st.secrets["FINNHUB_API_KEY"]
        
        # Clean the ticker symbol (e.g., FCIT.L -> FCIT)
        clean_symbol = ticker_symbol.split('.')[0]
        
        # FIX: Finnhub /company-news requires specific 'from' and 'to' date parameters
        today = datetime.today().strftime('%Y-%m-%d')
        thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={thirty_days_ago}&to={today}&token={api_key}"
        response = requests.get(url)
        news_list = response.json()
        
        if not news_list or not isinstance(news_list, list):
            return 0.0, "NEUTRAL", f"No recent headlines found for {clean_symbol} in last 30 days."
        
        scores = []
        headlines = []
        
        # Pull top 5 recent headlines
        for item in news_list[:5]:
            headline = item.get('headline', '')
            if headline:
                headlines.append(headline)
                vs = analyzer.polarity_scores(headline)
                scores.append(vs['compound'])
                
        if not scores:
            return 0.0, "NEUTRAL", "No textual data found in payload."
            
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.05:
            sentiment_label = "POSITIVE"
        elif avg_score <= -0.05:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
            
        return round(avg_score, 2), sentiment_label, headlines[0]
        
    except Exception as e:
        return 0.0, "NEUTRAL", f"API Error: {str(e)}"

# ------------------------------------
# FETCH PRICE HISTORY
# ------------------------------------
def get_price_change(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="2mo", interval="1d", group_by="column", progress=False)
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
# MAIN ENGINE SCANNER
# ------------------------------------
watch_list = ["IWM", "IJR", "AAPL", "NVDA", "TSLA", "MSFT"]

if st.button("🚀 Run Narrative Radar Scan"):
    results = []
    
    with st.spinner("Accessing Finnhub global terminal and analyzing sentiment..."):
        for ticker in watch_list:
            price_change = get_price_change(ticker)
            if price_change is None:
                continue
                
            sentiment_score, sentiment_label, top_news = get_ticker_sentiment(ticker)
            
            # Formulate Strategy Category based on quadrant rules
            if sentiment_score >= 0.05 and price_change > 0:
                action = "🚀 Momentum Leader (Buy/Hold)"
            elif sentiment_score >= 0.05 and price_change <= 0:
                action = "🔍 Bullish Divergence (Watch for Dip)"
            elif sentiment_score < -0.05 and price_change > 0:
                action = "⚠️ Bearish Divergence (Risk Flag)"
            elif sentiment_score <= -0.05 and price_change <= 0:
                action = "📉 Value Trap / Weak (Avoid)"
            else:
                action = "😐 Neutral / Rangebound"
                
            results.append({
                "Ticker": ticker,
                "30d Price Change (%)": price_change,
                "Sentiment Score (-1 to +1)": sentiment_score,
                "Narrative Status": sentiment_label,
                "Strategy / Action": action,
                "Latest Headline": top_news
            })
            
    if results:
        df_results = pd.DataFrame(results)
        
        st.write("### 📌 Tactical Market Screener")
        st.dataframe(df_results, use_container_width=True)
        
        st.write("### 🗺️ Narrative Matrix Visualization")
        fig = px.scatter(
            df_results, 
            x="Sentiment Score (-1 to +1)", 
            y="30d Price Change (%)",
            text="Ticker",
            color="Strategy / Action",
            range_x=[-1, 1],
            title="Where do your Tickers sit?",
            labels={"x": "News Sentiment Score", "y": "30-Day Price Trend (%)"}
        )
        fig.update_traces(textposition='top center', marker=dict(size=14))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not process any tickers. Check your API settings.")
