"""
mizan-ai - Search Agent
Web arama işlemlerini yöneten özel agent

SearchStrategyAgent ile entegre çalışır:
1. StrategyAgent sorguyu analiz eder, arama sorgularını üretir
2. SearchAgent bu sorguları çalıştırır ve sonuçları birleştirir
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.duckduckgo_search import DuckDuckGoSearch, SearchResult
from core.search_strategy_agent import get_search_strategy_agent, SearchStrategy

logger = logging.getLogger(__name__)


@dataclass
class SearchContext:
    """Arama sonuçlarını ve bağlamını tutar"""
    query: str
    party: Optional[str]
    results: List[SearchResult]
    formatted_text: str
    has_results: bool


class SearchAgent:
    """
    Web arama işlemlerini yöneten agent.

    Görevleri:
    1. Sorguyu analiz et
    2. Uygun arama stratejisi seç
    3. Sonuçları formatla
    4. Bağlam oluştur
    """

    # Soru ifadelerini temizlemek için pattern'ler
    QUESTION_PATTERNS = [
        r'\?$',                         # Soru işareti
        r'\blisteler?\s+misin\b',       # listeler misin
        r'\bsayar?\s+mısın\b',          # sayar mısın
        r'\banlatır?\s+mısın\b',        # anlatır mısın
        r'\bsöyler?\s+misin\b',         # söyler misin
        r'\bne(?:dir|lerdir)?\b',       # nedir, nelerdir
        r'\bkim(?:dir|lerdir)?\b',      # kimdir, kimlerdir
        r'\bnasıl\b',                   # nasıl
        r'\bhangi(?:si|leri)?\b',       # hangisi, hangileri
        r'\bmisin\b|\bmısın\b',         # misin, mısın
    ]

    def __init__(self, max_results: int = 5):
        self.search_engine = DuckDuckGoSearch(max_results=max_results)
        self.max_results = max_results

    def _clean_query_for_search(self, query: str) -> str:
        """Sorguyu arama için temizler - soru ifadelerini kaldırır."""
        import re

        cleaned = query.strip()

        # Soru pattern'lerini temizle
        for pattern in self.QUESTION_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Fazla boşlukları temizle
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned
    
    def search(self, query: str, party: Optional[str] = None) -> SearchContext:
        """
        SearchStrategyAgent ile entegre arama yapar.

        1. StrategyAgent sorguyu analiz eder
        2. Optimize edilmiş arama sorguları üretir
        3. Sorgular çalıştırılır ve sonuçlar birleştirilir

        Args:
            query: Arama sorgusu
            party: Parti kodu (opsiyonel)

        Returns:
            SearchContext: Arama bağlamı
        """
        logger.info(f"SearchAgent: '{query}' icin arama yapiliyor...")

        # 1. Strateji agent'ından sorguları al
        strategy_agent = get_search_strategy_agent()
        strategy = strategy_agent.create_strategy(query, party)

        logger.info(f"Strateji: {strategy.reasoning}")

        # 2. Tüm sorguları çalıştır ve sonuçları topla
        all_results: List[SearchResult] = []
        seen_urls = set()

        for search_query in strategy.search_queries:
            logger.info(f"Arama: '{search_query.query}'")

            # Arama yap
            results = self.search_engine.search(search_query.query, max_results=3)

            # Duplikasyonları önle
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

            # Yeterli sonuç varsa dur
            if len(all_results) >= self.max_results:
                break

        # 3. Alakasız sonuçları filtrele
        filtered_results = self._filter_results(all_results, party)

        # Sonuçları formatla
        formatted_text = self._format_results(filtered_results[:self.max_results])

        context = SearchContext(
            query=query,
            party=party,
            results=filtered_results[:self.max_results],
            formatted_text=formatted_text,
            has_results=len(filtered_results) > 0
        )

        logger.info(f"SearchAgent: {len(filtered_results)} alakali sonuc bulundu")
        return context
    
    def _search_party_specific(self, query: str, party: str) -> List[SearchResult]:
        """Parti-spesifik arama yapar"""
        # Sorguyu temizle
        cleaned_query = self._clean_query_for_search(query)

        # Gereksiz kelimeleri çıkar
        stop_words = {'ait', 'olan', 'partiye', 'partinin', 'partisi', 'parti'}
        cleaned_lower = cleaned_query.lower()

        # Parti adını ve stop words'leri çıkar
        words = cleaned_lower.split()
        keywords = [w for w in words if w not in stop_words and w.lower() != party.lower() and len(w) > 2]

        # Türkçe çoğul eklerini kaldır (basit)
        topic_words = []
        for w in keywords:
            # -leri, -ları, -lari vb. eklerini kaldır
            if w.endswith('leri') or w.endswith('lari') or w.endswith('ları'):
                topic_words.append(w[:-4])
            elif w.endswith('ler') or w.endswith('lar'):
                topic_words.append(w[:-3])
            else:
                topic_words.append(w)

        topic_clean = ' '.join(topic_words) if topic_words else ''

        logger.info(f"Parti: {party}, Topic: '{topic_clean}'")

        # search_party_news kullan - daha iyi sonuç veriyor
        return self.search_engine.search_party_news(party, topic_clean)
    
    def _search_general(self, query: str) -> List[SearchResult]:
        """Genel arama yapar"""
        enhanced_query = f"{query} Türkiye siyaset"
        return self.search_engine.search(enhanced_query, max_results=self.max_results)
    
    def _is_relevant_result(self, result: SearchResult, party: Optional[str]) -> bool:
        """Sonucun alakalı olup olmadığını kontrol eder."""
        # Alakasız sonuç kalıpları
        irrelevant_patterns = [
            r'NASA',
            r'Digital Elevation',
            r'Terrain Model',
            r'GIS',
            r'知乎',  # Çince site
            r'Earthdata',
            r'satellite',
            r'topography',
        ]

        title_lower = result.title.lower()
        snippet_lower = result.snippet.lower()

        for pattern in irrelevant_patterns:
            if re.search(pattern, result.title, re.IGNORECASE):
                return False
            if re.search(pattern, result.snippet, re.IGNORECASE):
                return False

        # Türkçe içerik kontrolü - en az bir Türkçe kelime içermeli
        turkish_keywords = ['parti', 'belediye', 'seçim', 'başkan', 'türkiye', 'il', 'ilçe']
        has_turkish = any(kw in title_lower or kw in snippet_lower for kw in turkish_keywords)

        return has_turkish

    def _filter_results(self, results: List[SearchResult], party: Optional[str]) -> List[SearchResult]:
        """Alakasız sonuçları filtreler."""
        filtered = [r for r in results if self._is_relevant_result(r, party)]
        logger.info(f"Filtreleme: {len(results)} -> {len(filtered)} sonuc")
        return filtered

    def _format_results(self, results: List[SearchResult]) -> str:
        """Sonuçları LLM için formatlar"""
        if not results:
            return ""

        formatted = "GÜNCEL WEB BİLGİLERİ:\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. {r.title}\n"
            formatted += f"   {r.snippet}\n"
            formatted += f"   Kaynak: {r.url}\n\n"

        return formatted
    
    def should_use_web_search(self, query: str, local_score: float) -> bool:
        """
        Web araması yapılması gerekip gerekmediğini belirler.
        
        Args:
            query: Kullanıcı sorgusu
            local_score: Yerel arama skoru
            
        Returns:
            bool: Web araması yapılsın mı?
        """
        # Düşük yerel skor = web araması gerekli
        if local_score < 0.5:
            return True
        
        # Zaman/tarih içeren sorular
        time_keywords = ["kimdir", "bugün", "son", "güncel", "ne zaman", "kaç", "nerede"]
        query_lower = query.lower()
        
        for keyword in time_keywords:
            if keyword in query_lower:
                return True
        
        return False


def create_search_agent() -> SearchAgent:
    """SearchAgent oluşturur"""
    return SearchAgent()
