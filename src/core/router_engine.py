"""
mizan-ai - Router Engine
Niyet Analizi & Karar Mekanizması
Sorguyu analiz ederek yerel bilgi veya web araması kararı verir.
"""

import re
import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentAnalysis:
    intent_type: str
    confidence: float
    needs_web_search: bool
    parties_mentioned: List[str]
    reasoning: str


class RouterEngine:
    """
    Sorgu yönlendirme motoru - Niyet analizi yapar.
    
    map.md mimarisine göre:
    - Tüzük/politika odaklı sorgular → Vektör Arama (RAG)
    - Güncel olay/gündem sorguları → Web Arama (DuckDuckGo)
    - Hibrit sorgular → Her ikisi de kullanılır
    """
    
    PARTI_KEYWORDS = {
        "CHP": ["chp", "cumhuriyet halk", "kılıçdaroğlu", "özel", "chp'li"],
        "AKP": ["akp", "adalet kalkınma", "erdoğan", "ak partili", "akp'li"],
        "MHP": ["mhp", "milliyetçi hareket", "bahçeli", "mhp'li"],
        "İYİ": ["iyi parti", "akşener", "iyi'li", "i̇yi"],
        "DEM": ["dem", "halkların eşitlik", "tuncer", "dem'li", "demli"],
        "SP": ["saadet", "saadet partisi", "karaca", "sp'li"],
        "ZP": ["zafer", "zafer partisi", "üzmcü", "zp'li"],
        "BBP": ["büyük birlik", "bbp", "bbp'li"],
    }
    
    WEB_SEARCH_TRIGGERS = [
        r"\b(bugün|son|güncel|son\s+haber|son\s+gelişme)\b",
        r"\b(miting|kongre|toplantı|açıklama|basın)\b",
        r"\b(2024|2025|2026)\b(?!\s*(yılında|yıl)|parti|tüzük)",
        r"\b(anket|oy|seçim| sandık)\b",
        r"\b(nerede|ne zaman|kimdir|kimdir bugün|kaç yaşında|doğum)\b",
        r"\b(açıkl(adı|ama)|duyurdu|belirtti)\b",
        r"\b(kim\s+olsun|kim\b.*\b(seçildi|olacak|başkan))\b",
        r"(genel\s*başkan|başbakan|cumhurbaşkanı)\s*(kim|nedir)",
    ]
    
    LOCAL_SEARCH_TRIGGERS = [
        r"\b(tüzük|madde|karar|genel kurul)\b",
        r"\b(politika|ilkeler|program)\b",
        r"\b(parti\b.*tüzük|tüzük\b.*parti)\b",
        r"\b(yönetim|kurul|başkanlık)\b",
        r"\b(kuruluş|tarihçe|geçmiş)\b",
        r"\b(\d+\.?\d*\s*(madde|madde\s*no))\b",
    ]
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
    
    def analyze_intent(self, query: str) -> IntentAnalysis:
        """
        Sorguyu analiz eder ve niyet tipini belirler.
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            IntentAnalysis: Niyet analizi sonucu
        """
        query_lower = query.lower()
        
        parties = self._extract_parties(query_lower)
        web_score = self._calculate_web_score(query_lower)
        local_score = self._calculate_local_score(query_lower)
        
        needs_web = web_score >= self.threshold
        
        reasoning = self._build_reasoning(web_score, local_score, parties)
        
        return IntentAnalysis(
            intent_type="web" if needs_web else "local",
            confidence=max(web_score, local_score),
            needs_web_search=needs_web,
            parties_mentioned=parties,
            reasoning=reasoning,
        )
    
    def _extract_parties(self, query: str) -> List[str]:
        """Sorguda bahsedilen partileri bulur."""
        parties = []
        for party, keywords in self.PARTI_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    if party not in parties:
                        parties.append(party)
                    break
        return parties
    
    def _calculate_web_score(self, query: str) -> float:
        """Web araması ihtiyacını hesaplar."""
        score = 0.0
        for pattern in self.WEB_SEARCH_TRIGGERS:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.2
        return min(score, 1.0)
    
    def _calculate_local_score(self, query: str) -> float:
        """Yerel vektör araması ihtiyacını hesaplar."""
        score = 0.0
        for pattern in self.LOCAL_SEARCH_TRIGGERS:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.25
        return min(score, 1.0)
    
    def _build_reasoning(self, web_score: float, local_score: float, parties: List[str]) -> str:
        """Karar sürecini açıklar."""
        reason_parts = []
        
        if parties:
            reason_parts.append(f"Partiler: {', '.join(parties)}")
        
        if web_score > local_score:
            reason_parts.append(f"Güncel olay tespiti: %{int(web_score*100)}")
        else:
            reason_parts.append(f"Tüzük/politika ağırlıklı: %{int(local_score*100)}")
        
        return " | ".join(reason_parts)
    
    def should_search_web(self, query: str) -> Tuple[bool, float]:
        """
        Sorgunun web araması gerektirip gerektirmediğini belirler.
        
        Returns:
            Tuple[bool, float]: (arama_gerekli, güven)
        """
        analysis = self.analyze_intent(query)
        return analysis.needs_web_search, analysis.confidence
    
    def get_parties_from_query(self, query: str) -> List[str]:
        """Sorgudan parti listesini çıkarır."""
        analysis = self.analyze_intent(query)
        return analysis.parties_mentioned


def create_router(threshold: float = 0.5) -> RouterEngine:
    """Router engine oluşturur."""
    return RouterEngine(threshold=threshold)
