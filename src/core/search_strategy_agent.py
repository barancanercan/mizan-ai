"""
Search Strategy Agent - Arama stratejisi belirleme ve sorgu optimizasyonu.

Bu agent:
1. Kullanıcı sorgusunu analiz eder
2. Ne tür bilgi arandığını tespit eder
3. Optimum arama sorgularını üretir
4. Arama sonuçlarını değerlendirir
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SearchIntent(Enum):
    """Arama niyeti türleri."""
    PERSON_CURRENT = "guncel_kisi"      # Şu anki başkan kim?
    PERSON_LIST = "kisi_listesi"        # Milletvekilleri kimler?
    ENTITY_LIST = "varlik_listesi"      # Belediyeler, iller, vs.
    EVENT_INFO = "olay_bilgisi"         # Kongre ne zaman?
    PROCESS_INFO = "surec_bilgisi"      # Nasıl seçilir?
    STATISTIC = "istatistik"            # Kaç üye var?
    COMPARISON = "karsilastirma"        # Farkları neler?
    GENERAL = "genel"                   # Genel bilgi


@dataclass
class SearchQuery:
    """Üretilen arama sorgusu."""
    query: str
    priority: int  # 1 = en yüksek öncelik
    intent: SearchIntent
    expected_keywords: List[str]  # Sonuçlarda beklenen kelimeler


@dataclass
class SearchStrategy:
    """Arama stratejisi."""
    original_query: str
    intent: SearchIntent
    search_queries: List[SearchQuery]
    party: Optional[str]
    reasoning: str  # Neden bu strateji seçildi


class SearchStrategyAgent:
    """
    Arama stratejisi belirleyen agent.

    Sorguyu analiz eder, ne tür bilgi arandığını anlar ve
    optimum arama sorgularını üretir.
    """

    # Niyet tespit kalıpları
    INTENT_PATTERNS = {
        SearchIntent.PERSON_CURRENT: [
            r'(?:genel\s*)?başkan[ıi]?\s*kim',
            r'kim(?:dir)?\s*(?:şu\s*an|mevcut|güncel)',
            r'lider[i]?\s*kim',
            r'yönetiyor',
        ],
        SearchIntent.PERSON_LIST: [
            r'milletvekil(?:i|leri)',
            r'(?:üye|delege|aday)lar[ıi]?\s*kim',
            r'yönetim\s*kadro',
            r'isim(?:ler)?',
        ],
        SearchIntent.ENTITY_LIST: [
            r'belediye(?:ler|si|leri)',
            r'il(?:ler|leri)(?:\s|$)',
            r'(?:il|ilçe)\s*(?:örgüt|teşkilat)',
            r'listele',
            r'hangi\s*(?:il|şehir|belediye)',
        ],
        SearchIntent.EVENT_INFO: [
            r'(?:kongre|kurultay|toplantı)\s*(?:ne\s*zaman|tarih)',
            r'ne\s*zaman\s*(?:yapıl|olacak|düzenlen)',
            r'tarih[i]?',
        ],
        SearchIntent.PROCESS_INFO: [
            r'nasıl\s*(?:seçil|yapıl|belirlen|oluş)',
            r'(?:süreç|yöntem|prosedür)',
            r'ne\s*şekilde',
        ],
        SearchIntent.STATISTIC: [
            r'kaç\s*(?:kişi|üye|oy|delege|milletvekili|belediye)',
            r'sayı(?:sı)?',
            r'oran[ıi]?',
            r'yüzde',
        ],
        SearchIntent.COMPARISON: [
            r'fark(?:ı|ları)',
            r'karşılaştır',
            r'(?:ile|ve)\s*arasında',
            r'hangisi\s*daha',
        ],
    }

    # Parti-spesifik arama terimleri
    PARTY_SEARCH_TERMS = {
        "DEM": {
            "aliases": ["DEM Parti", "HDP", "Halkların Eşitlik ve Demokrasi Partisi"],
            "leaders": ["Tuncer Bakırhan", "Tülay Hatimoğulları"],
            "keywords": ["Kürt", "demokratik özerklik", "eş başkan"],
        },
        "CHP": {
            "aliases": ["CHP", "Cumhuriyet Halk Partisi"],
            "leaders": ["Özgür Özel"],
            "keywords": ["sosyal demokrasi", "Atatürk", "laiklik"],
        },
        "AKP": {
            "aliases": ["AK Parti", "AKP", "Adalet ve Kalkınma Partisi"],
            "leaders": ["Erdoğan"],
            "keywords": ["iktidar", "hükümet"],
        },
        "MHP": {
            "aliases": ["MHP", "Milliyetçi Hareket Partisi"],
            "leaders": ["Devlet Bahçeli"],
            "keywords": ["milliyetçi", "ülkücü"],
        },
        "İYİ": {
            "aliases": ["İYİ Parti", "IYI Parti"],
            "leaders": ["Müsavat Dervişoğlu"],
            "keywords": ["merkez sağ"],
        },
    }

    # Konu-spesifik arama terimleri
    TOPIC_SEARCH_TERMS = {
        "belediye": ["belediye başkanı", "yerel seçim", "belediye meclisi", "büyükşehir"],
        "milletvekili": ["TBMM", "meclis", "milletvekili listesi", "seçim"],
        "kongre": ["kongre", "kurultay", "delege", "genel başkan seçimi"],
        "üyelik": ["parti üyeliği", "kayıt", "üye sayısı"],
    }

    def __init__(self):
        pass

    def _detect_intent(self, query: str) -> SearchIntent:
        """Sorgunun niyetini tespit eder."""
        query_lower = query.lower()

        # Türkçe karakter normalizasyonu
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        query_normalized = query_lower.translate(tr_map)

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                pattern_normalized = pattern.translate(tr_map)
                if re.search(pattern, query_lower) or re.search(pattern_normalized, query_normalized):
                    return intent

        return SearchIntent.GENERAL

    def _extract_topic(self, query: str) -> Optional[str]:
        """Sorgudan konuyu çıkarır."""
        query_lower = query.lower()

        for topic in self.TOPIC_SEARCH_TERMS.keys():
            if topic in query_lower:
                return topic

        return None

    def _generate_search_queries(
        self,
        query: str,
        intent: SearchIntent,
        party: Optional[str],
        topic: Optional[str]
    ) -> List[SearchQuery]:
        """Arama sorgularını üretir."""
        queries = []

        party_info = self.PARTY_SEARCH_TERMS.get(party, {})
        party_aliases = party_info.get("aliases", [party] if party else [])
        topic_terms = self.TOPIC_SEARCH_TERMS.get(topic, [])

        # Intent'e göre sorgu üret
        if intent == SearchIntent.ENTITY_LIST:
            # Belediye/il listesi için özel sorgular
            if topic == "belediye":
                for alias in party_aliases[:2]:
                    queries.append(SearchQuery(
                        query=f"{alias} kazandığı belediyeler 2024 yerel seçim",
                        priority=1,
                        intent=intent,
                        expected_keywords=["belediye", "seçim", "kazandı"]
                    ))
                    queries.append(SearchQuery(
                        query=f"{alias} belediye başkanları listesi 2024",
                        priority=2,
                        intent=intent,
                        expected_keywords=["başkan", "belediye", "liste"]
                    ))

        elif intent == SearchIntent.PERSON_CURRENT:
            # Güncel kişi bilgisi için
            for alias in party_aliases[:2]:
                queries.append(SearchQuery(
                    query=f"{alias} genel başkan kim 2024",
                    priority=1,
                    intent=intent,
                    expected_keywords=["başkan", "genel"]
                ))

        elif intent == SearchIntent.PERSON_LIST:
            # Kişi listesi için
            for alias in party_aliases[:2]:
                queries.append(SearchQuery(
                    query=f"{alias} milletvekilleri listesi 2024 TBMM",
                    priority=1,
                    intent=intent,
                    expected_keywords=["milletvekili", "TBMM", "liste"]
                ))

        elif intent == SearchIntent.STATISTIC:
            # İstatistik için
            for alias in party_aliases[:2]:
                if "belediye" in query.lower():
                    queries.append(SearchQuery(
                        query=f"{alias} kaç belediye kazandı 2024",
                        priority=1,
                        intent=intent,
                        expected_keywords=["belediye", "kazandı", "sayı"]
                    ))
                elif "milletvekili" in query.lower():
                    queries.append(SearchQuery(
                        query=f"{alias} milletvekili sayısı TBMM 2024",
                        priority=1,
                        intent=intent,
                        expected_keywords=["milletvekili", "sayı"]
                    ))

        # Genel fallback sorgusu
        if not queries:
            clean_query = self._clean_query(query)
            for alias in party_aliases[:1]:
                queries.append(SearchQuery(
                    query=f"{alias} {clean_query} Türkiye 2024",
                    priority=3,
                    intent=intent,
                    expected_keywords=[]
                ))

        return queries

    def _clean_query(self, query: str) -> str:
        """Sorgudan gereksiz kelimeleri temizler."""
        # Soru kalıplarını temizle
        patterns_to_remove = [
            r'\?$',
            r'\blisteler?\s*misin\b',
            r'\bsayar?\s*mısın\b',
            r'\banlatır?\s*mısın\b',
            r'\bsöyler?\s*misin\b',
            r'\bnedir\b',
            r'\bkimdir\b',
            r'\bnasıl\b',
            r'\bmisin\b|\bmısın\b',
            r'\bpartiye\b',
            r'\bpartinin\b',
            r'\bait\b',
        ]

        result = query
        for pattern in patterns_to_remove:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        return re.sub(r'\s+', ' ', result).strip()

    def create_strategy(self, query: str, party: Optional[str] = None) -> SearchStrategy:
        """
        Sorgu için arama stratejisi oluşturur.

        Args:
            query: Kullanıcı sorgusu
            party: Parti kodu (opsiyonel)

        Returns:
            SearchStrategy: Arama stratejisi
        """
        # Niyeti tespit et
        intent = self._detect_intent(query)

        # Konuyu çıkar
        topic = self._extract_topic(query)

        # Arama sorgularını üret
        search_queries = self._generate_search_queries(query, intent, party, topic)

        # Stratejik açıklama oluştur
        reasoning = self._create_reasoning(intent, topic, party, len(search_queries))

        logger.info(f"SearchStrategy: intent={intent.value}, topic={topic}, queries={len(search_queries)}")
        for sq in search_queries:
            logger.info(f"  [{sq.priority}] {sq.query}")

        return SearchStrategy(
            original_query=query,
            intent=intent,
            search_queries=search_queries,
            party=party,
            reasoning=reasoning
        )

    def _create_reasoning(
        self,
        intent: SearchIntent,
        topic: Optional[str],
        party: Optional[str],
        query_count: int
    ) -> str:
        """Strateji açıklaması oluşturur."""
        parts = []

        intent_desc = {
            SearchIntent.PERSON_CURRENT: "güncel kişi bilgisi",
            SearchIntent.PERSON_LIST: "kişi listesi",
            SearchIntent.ENTITY_LIST: "varlık listesi (belediye/il)",
            SearchIntent.EVENT_INFO: "etkinlik/olay bilgisi",
            SearchIntent.PROCESS_INFO: "süreç/yöntem bilgisi",
            SearchIntent.STATISTIC: "istatistik/sayısal bilgi",
            SearchIntent.COMPARISON: "karşılaştırma",
            SearchIntent.GENERAL: "genel bilgi",
        }

        parts.append(f"Arama türü: {intent_desc.get(intent, 'genel')}")

        if topic:
            parts.append(f"Konu: {topic}")

        if party:
            parts.append(f"Parti: {party}")

        parts.append(f"{query_count} farklı arama sorgusu üretildi")

        return ". ".join(parts)


# Singleton
_strategy_agent = None


def get_search_strategy_agent() -> SearchStrategyAgent:
    """Singleton Search Strategy Agent döndürür."""
    global _strategy_agent
    if _strategy_agent is None:
        _strategy_agent = SearchStrategyAgent()
    return _strategy_agent