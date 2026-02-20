"""
mizan-ai - DuckDuckGo Web Search
Ücretsiz web arama modülü
Tavily yerine DuckDuckGo kullanarak güncel siyasi bilgileri çeker.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


class DuckDuckGoSearch:
    """
    DuckDuckGo üzerinden web araması yapar.
    
    map.md mimarisinde:
    - Web Arama (Tavily) yerine ücretsiz alternatif
    - Güncel siyasi haberler ve gelişmeler için kullanılır
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        DuckDuckGo'da arama yapar.
        
        Args:
            query: Arama sorgusu
            max_results: Maksimum sonuç sayısı
            
        Returns:
            List[SearchResult]: Arama sonuçları
        """
        try:
            from duckduckgo_search import DDGS  # pip install duckduckgo-search
            
            results = []
            max_res = max_results or self.max_results
            
            # Arama stratejisi: Önce news, sonra text
            with DDGS(timeout=self.timeout) as ddgs:
                # Önce haberleri dene
                try:
                    for r in ddgs.news(query, max_results=max_res):
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            snippet=r.get("body", ""),
                            source="duckduckgo-news"
                        ))
                except:
                    pass
                
                # Sonra normal arama
                if len(results) < max_res:
                    for r in ddgs.text(query, max_results=max_res - len(results)):
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            source="duckduckgo"
                        ))
            
            logger.info(f"✅ DuckDuckGo: {len(results)} sonuç bulundu")
            return results
            
        except ImportError:
            logger.warning("⚠️ duckduckgo-search kurulu değil")
            return self._mock_search(query)
        except Exception as e:
            logger.error(f"❌ DuckDuckGo arama hatası: {str(e)}")
            return self._mock_search(query)
    
    def _mock_search(self, query: str) -> List[SearchResult]:
        """Mock sonuçlar döner (test için)."""
        return [
            SearchResult(
                title=f"Mock Sonuç: {query}",
                url="https://example.com",
                snippet=f"Bu bir mock sonuçtur. duckduckgo-search paketini kurunuz.",
                source="mock"
            )
        ]
    
    def search_turkish_news(self, query: str, days: int = 7) -> List[SearchResult]:
        """
        Türkçe haberler için özelleştirilmiş arama.
        
        Args:
            query: Arama sorgusu
            days: Son kaç gün içinde
            
        Returns:
            List[SearchResult]: Haber sonuçları
        """
        enhanced_query = f"{query} Türkiye siyaset"
        return self.search(enhanced_query)
    
    def search_party_news(self, party: str, topic: str = "") -> List[SearchResult]:
        """
        Belirli bir parti ile ilgili haberleri arar.

        Args:
            party: Parti adı veya kodu
            topic: Konu (opsiyonel)

        Returns:
            List[SearchResult]: Haber sonuçları
        """
        party_names = {
            "CHP": "CHP Cumhuriyet Halk Partisi",
            "AKP": "AK Parti AKP",
            "MHP": "MHP Milliyetçi Hareket Partisi",
            "İYİ": "IYI Parti",
            "DEM": "DEM Parti HDP",  # HDP ile de ara - daha fazla sonuç
            "SP": "Saadet Partisi",
            "ZP": "Zafer Partisi",
            "BBP": "Buyuk Birlik Partisi BBP",
        }

        party_name = party_names.get(party, party)

        # Konu varsa ekle
        if topic:
            query = f"{party_name} {topic} 2024"
        else:
            query = f"{party_name} 2024"

        logger.info(f"Parti haberi sorgusu: {query}")
        return self.search(query)


def create_search_engine() -> DuckDuckGoSearch:
    """DuckDuckGo search engine oluşturur."""
    return DuckDuckGoSearch()


def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Web'de hızlı arama yapar.
    
    Args:
        query: Arama sorgusu
        max_results: Maksimum sonuç sayısı
        
    Returns:
        List[SearchResult]: Arama sonuçları
    """
    engine = create_search_engine()
    return engine.search(query, max_results)
