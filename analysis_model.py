# analysis_model.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class PricePoint:
    timestamp: datetime
    price: float
    volume: int = 0

@dataclass
class PriceSeries:
    symbol: str
    data: List[PricePoint] = field(default_factory=list)
    def get_range(self, start: datetime, end: datetime) -> List[PricePoint]:
        return [p for p in self.data if start <= p.timestamp <= end]

@dataclass
class Stock:
    symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def get_latest_price(self, series: PriceSeries) -> Optional[float]:
        pts = [p for p in series.data if p.timestamp]
        if not pts:
            return None
        return sorted(pts, key=lambda x: x.timestamp)[-1].price

@dataclass
class NewsArticle:
    article_id: str
    title: str
    content: str
    source: str
    publish_date: datetime
    symbols: List[str] = field(default_factory=list)

@dataclass
class SentimentResult:
    article_id: str
    score: float  # -1..1
    label: str    # negative/neutral/positive
    confidence: float = 1.0

@dataclass
class PortfolioItem:
    symbol: str
    quantity: int

@dataclass
class Portfolio:
    owner_id: str
    items: List[PortfolioItem] = field(default_factory=list)

    def total_value(self, price_lookup: Dict[str, float]) -> float:
        total = 0.0
        for it in self.items:
            price = price_lookup.get(it.symbol)
            if price is None:
                continue
            total += price * it.quantity
        return total

# Lightweight interfaces for services (analysis level)
class DataFetcher:
    def fetch_price(self, symbol: str, start: datetime, end: datetime) -> PriceSeries:
        raise NotImplementedError
    def fetch_news(self, symbol: str, since: datetime) -> List[NewsArticle]:
        raise NotImplementedError

class SentimentAnalyzer:
    def analyze(self, article: NewsArticle) -> SentimentResult:
        raise NotImplementedError
