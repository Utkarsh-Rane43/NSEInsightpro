from transformers import pipeline

# Download 'distilbert-base-uncased-finetuned-sst-2-english' once
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    res = sentiment_pipeline(text)[0]
    label = res['label'].lower()
    if "positive" in label:
        return "positive"
    elif "negative" in label:
        return "negative"
    else:
        return "neutral"
