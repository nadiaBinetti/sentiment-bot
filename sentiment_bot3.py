# Generating the full Python script including NewsAPI, Twitter, StockTwits, Reddit sentiment analysis,
# with FinBERT and RoBERTa models, and Telegram integration.

import os
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import praw

# ========== CONFIGURAZIONE ==========
# Telegram bot
BOT_TOKEN = os.environ.get("BOT_TOKEN") # Sostituisci con il nome esatto del tuo secret su GitHub
CHAT_ID = os.environ.get("CHAT_ID")     # Sostituisci con il nome esatto del tuo secret su GitHub

#TWITTER
TWITTER_BEARER_TOKEN = os.environ.get("BEARER_TOKEN") # E così via per tutti...

#NEWAPI
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

# Reddit
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")


# Tickers
TICKERS = ["AAPL", "NVDA", "MSFT","SNAP","GOOG","AMZN","META","PST.MI","KO"]
#TICKERS = ["TSLA"]


#PESI
NEWS_WEIGHT=0.6
TWITTER_WEIGTH=0.1
REDDIT_WEIGHT=0.1
STOCK_WEIGHT=0.2

# ========== TELEGRAM ==========

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

# ========== MODELLI ==========

# FinBERT per news
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_labels = ["Negative", "Neutral", "Positive"]

def analyze_news_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    label = finbert_labels[torch.argmax(probs)]
    return label

# RoBERTa per social
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_labels = ["Negative", "Neutral", "Positive"]

def analyze_social_sentiment(posts):
    inputs = roberta_tokenizer(posts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    results = []
    for i, p in enumerate(probs):
        label = roberta_labels[torch.argmax(p)]
        score = torch.max(p).item()
        results.append((posts[i], label, round(score, 2)))
    return results

# ========== API DATI ==========

def get_latest_news(ticker, max_articles=3):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize={max_articles}&language=en"
    try:
        response = requests.get(url)
        data = response.json()
        return [article["title"] for article in data.get("articles", [])]
    except Exception as e:
        print(f"Errore NewsAPI per {ticker}: {e}")
        return []

def get_twitter_posts(ticker):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
    }
    params = {
        "query": f"${ticker} -is:retweet lang:en",
        "max_results": 10,
        "tweet.fields": "created_at,text,lang"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Errore Twitter API per {ticker}: {response.json()}")
            return []
        data = response.json()
        tweets = [tweet["text"] for tweet in data.get("data", [])]
        return tweets
    except Exception as e:
        print(f"Errore nella chiamata Twitter per {ticker}: {e}")
        return []

def get_stocktwits_posts(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"}
        response = requests.get(url, headers=headers)
   
        #response = requests.get(url)
        if response.status_code != 200:
            print(f"Errore StockTwits API per {ticker}: HTTP {response.status_code}")
            return []
        try:
            data = response.json()
        except Exception as json_error:
            print(f"Errore di parsing JSON per {ticker}: {json_error}")
            print(f"Risposta grezza: {response.text}")
            return []
        return [m["body"] for m in data.get("messages", [])[:5]]
    except Exception as e:
        print(f"Errore StockTwits API per {ticker}: {e}")
        return []

def get_reddit_posts(ticker):
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                             client_secret=REDDIT_CLIENT_SECRET,
                             user_agent=REDDIT_USER_AGENT)
        subreddit = reddit.subreddit("wallstreetbets")
        posts = []
        for submission in subreddit.search(ticker, limit=5):
            posts.append(submission.title)
        return posts
    except Exception as e:
        print(f"Errore Reddit API per {ticker}: {e}")
        return []

# ========== COMBINAZIONE ==========

def sentiment_to_score(label):
    return {"Positive": 1, "Neutral": 0, "Negative": -1}.get(label, 0)

def combine_sentiments(news_sentiment, twitter_results, stocktwits_results, reddit_results):
    news_score = sentiment_to_score(news_sentiment)
    twitter_score = sum([sentiment_to_score(r[1]) for r in twitter_results]) / len(twitter_results) if twitter_results else 0
    stocktwits_score = sum([sentiment_to_score(r[1]) for r in stocktwits_results]) / len(stocktwits_results) if stocktwits_results else 0
    reddit_score = sum([sentiment_to_score(r[1]) for r in reddit_results]) / len(reddit_results) if reddit_results else 0
    total_score = (NEWS_WEIGHT * news_score) + (TWITTER_WEIGTH * twitter_score) + (STOCK_WEIGHT * stocktwits_score) + (REDDIT_WEIGHT * reddit_score)
    if total_score > 0.25:
        return "BUY"
    elif total_score < -0.25:
        return "SELL"
    else:
        return "HOLD"

# ========== MAIN ==========

if __name__ == "__main__":
    if not BOT_TOKEN or not CHAT_ID: # Aggiungi tutti i tuoi secret qui
    print("Errore: Uno o più secret non sono stati caricati correttamente dalle variabili d'ambiente.")
    exit(1) # Esce dallo script se mancano i secret
    
    full_report = "📊 Verdetto Giornaliero:\n\n"
    for ticker in TICKERS:
        print(f"Analizzando {ticker}...")

        news_articles = get_latest_news(ticker)
        news_text = " ".join(news_articles) if news_articles else "No recent news."
        news_sentiment = analyze_news_sentiment(news_text)

        twitter_posts = get_twitter_posts(ticker)
        twitter_results = analyze_social_sentiment(twitter_posts) if twitter_posts else []

        stocktwits_posts = get_stocktwits_posts(ticker)
        stocktwits_results = analyze_social_sentiment(stocktwits_posts) if stocktwits_posts else []

        reddit_posts = get_reddit_posts(ticker)
        reddit_results = analyze_social_sentiment(reddit_posts) if reddit_posts else []

        final_verdict = combine_sentiments(news_sentiment, twitter_results, stocktwits_results, reddit_results)

        report = (
            f"📈 {ticker}\n"
            f"📰 News: {news_sentiment}\n"
            f"🐦 Twitter: {[r[1] for r in twitter_results]}\n"
            f"📊 StockTwits: {[r[1] for r in stocktwits_results]}\n"
            f"👽 Reddit: {[r[1] for r in reddit_results]}\n"
            f"✅ Verdetto: {final_verdict}\n\n"
        )
        full_report += report

    print(full_report)
    send_telegram_message(full_report)
    
    
    # ======= TRENDING TICKERS AGGIUNTIVI =======

    def get_trending_tickers_stocktwits(limit=10):
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Errore StockTwits API trending: HTTP {response.status_code}")
                return []
            data = response.json()
            symbols = data.get("symbols", [])
            # Escludi crypto
            filtered_symbols = [s["symbol"] for s in symbols if s.get("exchange") != "CRYPTO"]
            return filtered_symbols[:10]  # Limita ai primi 10 trending non-crypto
        except Exception as e:
            print(f"Errore nel recupero trending da StockTwits: {e}")
            return []

    # Analisi dei trending ticker
    trending_report = "🔥 Titoli più chiacchierati ora:\n\n"
    TRENDING_TICKERS = get_trending_tickers_stocktwits(limit=5)

    for ticker in TRENDING_TICKERS:
        print(f"[TRENDING] Analizzando {ticker}...")

        news_articles = get_latest_news(ticker)
        news_text = " ".join(news_articles) if news_articles else "No recent news."
        news_sentiment = analyze_news_sentiment(news_text)

        twitter_posts = get_twitter_posts(ticker)
        twitter_results = analyze_social_sentiment(twitter_posts) if twitter_posts else []

        stocktwits_posts = get_stocktwits_posts(ticker)
        stocktwits_results = analyze_social_sentiment(stocktwits_posts) if stocktwits_posts else []

        reddit_posts = get_reddit_posts(ticker)
        reddit_results = analyze_social_sentiment(reddit_posts) if reddit_posts else []

        final_verdict = combine_sentiments(news_sentiment, twitter_results, stocktwits_results, reddit_results)

        report = (
            f"📈 {ticker}\n"
            f"📰 News: {news_sentiment}\n"
            f"🐦 Twitter: {[r[1] for r in twitter_results]}\n"
            f"📊 StockTwits: {[r[1] for r in stocktwits_results]}\n"
            f"👽 Reddit: {[r[1] for r in reddit_results]}\n"
            f"✅ Verdetto: {final_verdict}\n\n"
        )
        trending_report += report

    send_telegram_message(trending_report)

