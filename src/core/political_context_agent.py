"""
Political Context Agent - Siyasi bağlam analizi ve anahtar kelime üretimi.

Bu agent:
1. Sorguları siyasi bağlamda analiz eder
2. Parti algılama doğruluğunu artırır (DEM vs DP gibi)
3. Retrieval için optimize edilmiş anahtar kelimeler üretir
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Parti tam adları ve kısaltmaları
PARTY_MAPPINGS: Dict[str, Dict] = {
    "DEM": {
        "full_names": ["halkların eşitlik ve demokrasi partisi", "dem parti", "dem partisi"],
        "keywords": ["eş genel başkan", "hdp", "demirtaş", "buldan", "kürt", "halkların"],
        "retrieval_terms": {
            "genel başkan": "Eş Genel Başkanlar Büyük Kongre delege seçim",
            "seçim": "kongre delege oy seçim gizli oy",
            "yönetim": "parti meclisi merkez yürütme kurulu myk",
        },
        "not_confused_with": ["demokrat parti", "dp"],
    },
    "DP": {
        "full_names": ["demokrat parti"],
        "keywords": ["menderes", "bayar", "demokrat"],
        "not_confused_with": ["dem parti", "dem"],
    },
    "AKP": {
        "full_names": ["adalet ve kalkınma partisi", "ak parti", "akp"],
        "keywords": ["erdoğan", "genel başkan", "mkyk", "kongre"],
        "retrieval_terms": {
            "genel başkan": "Genel Başkan Büyük Kongre delege seçim oy",
            "seçim": "kongre delege seçim oy oylama",
            "yönetim": "merkez karar yönetim kurulu mkyk genel idare",
        },
        "not_confused_with": [],
    },
    "CHP": {
        "full_names": ["cumhuriyet halk partisi", "chp"],
        "keywords": ["atatürk", "kılıçdaroğlu", "özel", "pm", "kurultay"],
        "retrieval_terms": {
            "genel başkan": "Genel Başkan kurultay delege seçim oy",
            "seçim": "kurultay delege seçim oy oylama",
            "yönetim": "parti meclisi pm merkez yürütme kurulu",
        },
        "not_confused_with": [],
    },
    "MHP": {
        "full_names": ["milliyetçi hareket partisi", "mhp"],
        "keywords": ["bahçeli", "ülkücü", "bozkurt"],
        "retrieval_terms": {
            "genel başkan": "Genel Başkan Büyük Kongre delege seçim",
            "seçim": "kongre delege seçim oy",
            "yönetim": "merkez yönetim kurulu genel idare kurulu",
        },
        "not_confused_with": [],
    },
    "İYİ": {
        "full_names": ["iyi parti", "iyiparti"],
        "keywords": ["akşener", "meral"],
        "retrieval_terms": {
            "genel başkan": "Genel Başkan Büyük Kongre delege seçim",
            "seçim": "kongre delege seçim oy",
            "yönetim": "genel idare kurulu gik parti meclisi",
        },
        "not_confused_with": [],
    },
    "SP": {
        "full_names": ["saadet partisi", "sp"],
        "keywords": ["erbakan", "karamollaoğlu", "milli görüş"],
        "not_confused_with": [],
    },
    "BBP": {
        "full_names": ["büyük birlik partisi", "bbp"],
        "keywords": ["muhsin yazıcıoğlu", "destici"],
        "not_confused_with": [],
    },
    "ZP": {
        "full_names": ["zafer partisi", "zp"],
        "keywords": ["ümit özdağ", "göçmen"],
        "not_confused_with": [],
    },
}

# Siyasi terimler sözlüğü (retrieval için genişletme)
POLITICAL_TERMS: Dict[str, List[str]] = {
    "genel başkan": ["genel başkan", "eş genel başkan", "parti lideri", "başkan", "lider"],
    "seçim": ["seçim", "seçilme", "seçilir", "oy", "oylama", "kongre", "kurultay"],
    "tüzük": ["tüzük", "parti tüzüğü", "yönetmelik", "madde", "kural"],
    "üyelik": ["üye", "üyelik", "kayıt", "parti üyesi", "delege"],
    "yönetim": ["yönetim", "yönetim kurulu", "mkyk", "myk", "pm", "parti meclisi"],
    "disiplin": ["disiplin", "ceza", "ihraç", "uyarı", "disiplin kurulu"],
}


@dataclass
class PoliticalContext:
    """Siyasi bağlam analizi sonucu."""
    detected_party: Optional[str]
    confidence: float
    expanded_keywords: List[str]
    disambiguation_note: Optional[str]
    original_query: str


class PoliticalContextAgent:
    """Siyasi sorguları analiz eden ve zenginleştiren agent."""

    def __init__(self):
        self.party_mappings = PARTY_MAPPINGS
        self.political_terms = POLITICAL_TERMS

    def detect_party(self, query: str) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Sorgudaki parti referansını tespit eder.

        Returns:
            Tuple[party_code, confidence, disambiguation_note]
        """
        query_lower = query.lower()
        detected_parties = []

        for party_code, info in self.party_mappings.items():
            score = 0.0

            # Tam isim eşleşmesi (en yüksek güven)
            for full_name in info["full_names"]:
                if full_name in query_lower:
                    score = max(score, 0.95)

            # Anahtar kelime eşleşmesi
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    score = max(score, 0.7)

            # Kısa kod eşleşmesi (word boundary ile)
            pattern = rf'\b{party_code.lower()}\b'
            if re.search(pattern, query_lower):
                score = max(score, 0.8)

            if score > 0:
                detected_parties.append((party_code, score))

        if not detected_parties:
            return None, 0.0, None

        # En yüksek skorlu partiyi seç
        detected_parties.sort(key=lambda x: x[1], reverse=True)
        best_party, best_score = detected_parties[0]

        # Belirsizlik kontrolü (DEM vs DP gibi)
        disambiguation_note = None
        if best_party in ["DEM", "DP"]:
            # Karıştırılabilecek terimleri kontrol et
            not_confused = self.party_mappings[best_party]["not_confused_with"]
            for term in not_confused:
                if term in query_lower:
                    # Yanlış parti algılanmış olabilir
                    alternative = "DP" if best_party == "DEM" else "DEM"
                    disambiguation_note = f"Not: '{term}' ifadesi {alternative} Partisi'ni işaret edebilir."
                    best_score *= 0.7  # Güveni düşür

        return best_party, best_score, disambiguation_note

    def expand_query_keywords(self, query: str) -> List[str]:
        """Sorguyu siyasi terimlerle genişletir."""
        query_lower = query.lower()
        expanded = []

        for term, expansions in self.political_terms.items():
            if term in query_lower:
                expanded.extend(expansions)

        # Duplikasyonları kaldır
        return list(set(expanded))

    def analyze(self, query: str, selected_party: Optional[str] = None) -> PoliticalContext:
        """
        Sorguyu analiz eder ve siyasi bağlam çıkarır.

        Args:
            query: Kullanıcı sorgusu
            selected_party: UI'da seçilen parti (varsa)

        Returns:
            PoliticalContext: Analiz sonucu
        """
        detected_party, confidence, disambiguation = self.detect_party(query)

        # Eğer UI'da parti seçilmişse, onu öncelikle kullan
        if selected_party:
            detected_party = selected_party
            confidence = 1.0
            disambiguation = None

        expanded_keywords = self.expand_query_keywords(query)

        logger.info(f"PoliticalContext: party={detected_party}, conf={confidence:.2f}, keywords={len(expanded_keywords)}")

        return PoliticalContext(
            detected_party=detected_party,
            confidence=confidence,
            expanded_keywords=expanded_keywords,
            disambiguation_note=disambiguation,
            original_query=query,
        )

    def _normalize_turkish(self, text: str) -> str:
        """Türkçe karakterleri ASCII'ye çevirir (eşleştirme için)."""
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        return text.translate(tr_map)

    def enhance_query_for_retrieval(self, query: str, party: str) -> str:
        """
        Retrieval için optimize edilmiş sorgu oluşturur.
        Parti-spesifik terminoloji kullanır.

        Args:
            query: Orijinal sorgu
            party: Hedef parti

        Returns:
            str: Genişletilmiş sorgu
        """
        query_lower = query.lower()
        query_normalized = self._normalize_turkish(query_lower)
        party_info = self.party_mappings.get(party, {})

        enhanced_parts = []

        # Parti-spesifik retrieval terimleri varsa kullan (normalize edilmiş eşleşme)
        retrieval_terms = party_info.get("retrieval_terms", {})
        for trigger, expansion in retrieval_terms.items():
            trigger_normalized = self._normalize_turkish(trigger.lower())
            if trigger_normalized in query_normalized:
                enhanced_parts.append(expansion)
                logger.info(f"Retrieval term matched: '{trigger}' -> '{expansion}'")

        # Eğer parti-spesifik terim bulunamadıysa, genel genişletme yap
        if not enhanced_parts:
            enhanced_parts.append(query)
            full_names = party_info.get("full_names", [])
            if full_names:
                enhanced_parts.append(full_names[0])

            # Genel anahtar kelimeler
            context = self.analyze(query, party)
            if context.expanded_keywords:
                enhanced_parts.extend(context.expanded_keywords[:3])
        else:
            # Parti-spesifik terim bulundu, orijinal sorguyu da ekle
            enhanced_parts.insert(0, query)

        return " ".join(enhanced_parts)


# Singleton instance
_agent_instance = None


def get_political_agent() -> PoliticalContextAgent:
    """Singleton Political Context Agent döndürür."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PoliticalContextAgent()
    return _agent_instance
