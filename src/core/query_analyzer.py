"""
Query Analyzer Agent - Sorgu analizi ve alt sorulara ayırma.

Bu agent:
1. Sorguyu analiz eder ve alt sorulara ayırır
2. Her alt soru için ayrı retrieval stratejisi belirler
3. Cevap tamamlığını kontrol eder
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Soru tipi."""
    WHO = "kim"           # Kim, kimdir, kimler
    WHAT = "ne"           # Ne, nedir, neler
    HOW = "nasıl"         # Nasıl, ne şekilde
    WHEN = "ne zaman"     # Ne zaman, hangi tarihte
    WHERE = "nerede"      # Nerede, hangi yerde
    WHY = "neden"         # Neden, niçin
    HOW_MANY = "kaç"      # Kaç, kaç tane
    WHICH = "hangi"       # Hangi, hangisi
    UNKNOWN = "bilinmiyor"


@dataclass
class SubQuestion:
    """Alt soru."""
    text: str
    question_type: QuestionType
    keywords: List[str]
    requires_web: bool = False  # Güncel bilgi gerektirir mi?
    answered: bool = False
    answer: Optional[str] = None


@dataclass
class QueryAnalysis:
    """Sorgu analizi sonucu."""
    original_query: str
    sub_questions: List[SubQuestion]
    is_compound: bool  # Birden fazla soru içeriyor mu?
    party_context: Optional[str] = None


class QueryAnalyzer:
    """Sorguları analiz eden ve alt sorulara ayıran agent."""

    # Türkçe karakter normalizasyonu için
    TR_CHAR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

    @staticmethod
    def _normalize_turkish(text: str) -> str:
        """Türkçe karakterleri ASCII'ye çevirir."""
        return text.translate(QueryAnalyzer.TR_CHAR_MAP)

    # Soru kalıpları (hem Türkçe hem ASCII versiyonları)
    QUESTION_PATTERNS = {
        QuestionType.WHO: [
            r'\bkimdir\b',              # "kimdir" - en spesifik önce
            r'\bkim(?:ler|lerdir)?\b',  # "kim", "kimler", "kimlerdir"
            r'\bkimin\b',
            r'başkan[ıi]?\s*kim',       # "başkanı kim", "başkan kim"
            r'lider[i]?\s*kim',         # "lideri kim"
            r'genel\s*başkan.*kim',     # "genel başkan kimdir"
        ],
        QuestionType.HOW: [
            r'\bnasıl\b',
            r'\bnasil\b',           # ASCII versiyonu
            r'\bne\s+şekilde\b',
            r'\bne\s+sekilde\b',    # ASCII versiyonu
            r'\bne\s+biçimde\b',
            r'\bhangi\s+yöntemle\b',
        ],
        QuestionType.WHAT: [
            r'\bnedir\b',
            r'\bne(?:ler)?\b',
            r'\bneleri\b',
            r'\blistele',               # "listeler misin"
            r'\bsay(?:ar|abilir)\b',    # "sayar mısın"
            r'\bhang[iı](?:si|leri)?\b', # "hangileri"
        ],
        QuestionType.WHEN: [
            r'\bne\s+zaman\b',
            r'\bhangi\s+tarih\b',
            r'\bkaç\s+yıl\b',
        ],
        QuestionType.HOW_MANY: [
            r'\bkaç\s+(?:kişi|üye|delege|oy)\b',
            r'\bkaç\s+tane\b',
        ],
        QuestionType.WHICH: [
            r'\bhangi(?:si|leri)?\b',
        ],
    }

    # Soru ayırıcı kalıplar
    QUESTION_SEPARATORS = [
        r'\?\s*',           # Soru işareti
        r'\s+ve\s+',        # "ve" bağlacı
        r'\s*,\s*',         # Virgül
        r'\s+ayrıca\s+',    # "ayrıca"
    ]

    # Güncel bilgi gerektiren kalıplar
    CURRENT_INFO_PATTERNS = [
        r'\bşu\s*an(?:ki|da)?\b',
        r'\bmevcut\b',
        r'\bgüncel\b',
        r'\bşimdiki\b',
        r'\bkim(?:dir)?\b',              # "kimdir" - güncel kişi sorusu
        r'\bbaşkan(?:ı)?\s+kim\b',       # "başkanı kim"
        r'\bgenel\s*başkan.*kim',        # "genel başkan kimdir"
        # Güncel durum/sahiplik soruları
        r'\bbelediye(?:ler|si|leri)?\b', # belediye soruları
        r'\bmilletvekil(?:i|leri)?\b',   # milletvekili soruları
        r'\bsayısı\b',                   # "kaç üyesi var" gibi
        r'\blistele',                    # "listeler misin"
        r'\bhang[iı]\s+(?:il|şehir|belediye)',  # "hangi iller"
        r'\bkazan[dıi]',                 # "kazandı" - seçim sonuçları
        r'\bson\s+(?:seçim|kongre)',     # "son seçim"
        r'\b20[12][0-9]\b',              # yıl referansı (2020-2029)
    ]

    def __init__(self):
        pass

    def _detect_question_type(self, text: str) -> QuestionType:
        """Soru tipini tespit eder."""
        text_lower = text.lower()
        text_normalized = self._normalize_turkish(text_lower)

        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                # Hem orijinal hem normalize edilmiş metinde ara
                if re.search(pattern, text_lower) or re.search(pattern, text_normalized):
                    return q_type

        return QuestionType.UNKNOWN

    def _requires_web_search(self, text: str) -> bool:
        """Güncel bilgi gerektirip gerektirmediğini kontrol eder."""
        text_lower = text.lower()
        text_normalized = self._normalize_turkish(text_lower)

        for pattern in self.CURRENT_INFO_PATTERNS:
            # Hem orijinal hem normalize edilmiş metinde ara
            if re.search(pattern, text_lower) or re.search(pattern, text_normalized):
                return True

        return False

    def _extract_keywords(self, text: str) -> List[str]:
        """Anahtar kelimeleri çıkarır."""
        # Stop words (Türkçe)
        stop_words = {
            'bir', 'bu', 'şu', 'o', 've', 'ile', 'için', 'de', 'da',
            'mi', 'mı', 'mu', 'mü', 'ne', 'nasıl', 'kim', 'hangi',
            'nedir', 'kimdir', 'olan', 'olarak', 'gibi', 'kadar',
        }

        # Kelimeleri ayır
        words = re.findall(r'\b\w+\b', text.lower())

        # Stop words'leri filtrele ve 2 karakterden uzun olanları al
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _split_compound_query(self, query: str) -> List[str]:
        """Birleşik sorguyu alt sorgulara ayırır."""
        # Önce soru işaretlerinden ayır
        parts = re.split(r'\?\s*', query)
        parts = [p.strip() for p in parts if p.strip()]

        # Her parçayı "ve", "ayrıca" gibi bağlaçlardan ayır
        sub_parts = []
        for part in parts:
            # "ve" bağlacından ayır - soru kelimesi geliyorsa
            # "kimdir ve nasıl" → ["kimdir", "nasıl seçilir"]
            sub = re.split(r'\s+ve\s+(?=(?:nasıl|kim|ne|hangi|nerede|ne\s+zaman|kaç)\b)', part, flags=re.IGNORECASE)
            sub_parts.extend([s.strip() for s in sub if s.strip()])

        # Eğer parça yoksa orijinal sorguyu döndür
        if not sub_parts:
            return [query]

        return sub_parts

    def analyze(self, query: str, party: Optional[str] = None) -> QueryAnalysis:
        """
        Sorguyu analiz eder.

        Args:
            query: Kullanıcı sorgusu
            party: Hedef parti (varsa)

        Returns:
            QueryAnalysis: Analiz sonucu
        """
        # Alt sorgulara ayır
        sub_query_texts = self._split_compound_query(query)

        sub_questions = []
        for text in sub_query_texts:
            q_type = self._detect_question_type(text)
            keywords = self._extract_keywords(text)
            requires_web = self._requires_web_search(text)

            sub_questions.append(SubQuestion(
                text=text,
                question_type=q_type,
                keywords=keywords,
                requires_web=requires_web,
            ))

        is_compound = len(sub_questions) > 1

        logger.info(f"QueryAnalyzer: {len(sub_questions)} alt soru tespit edildi, compound={is_compound}")
        for i, sq in enumerate(sub_questions):
            logger.info(f"  {i+1}. [{sq.question_type.value}] {sq.text[:50]}... web={sq.requires_web}")

        return QueryAnalysis(
            original_query=query,
            sub_questions=sub_questions,
            is_compound=is_compound,
            party_context=party,
        )

    def check_completeness(
        self,
        analysis: QueryAnalysis,
        answer: str,
    ) -> Tuple[bool, List[SubQuestion]]:
        """
        Cevabın tüm alt soruları kapsayıp kapsamadığını kontrol eder.

        Args:
            analysis: Sorgu analizi
            answer: Verilen cevap

        Returns:
            Tuple[bool, List[SubQuestion]]: (Tam mı?, Cevaplanmamış sorular)
        """
        answer_lower = answer.lower()
        unanswered = []

        for sq in analysis.sub_questions:
            # Basit keyword eşleştirmesi
            answered = False

            # WHO tipi sorular için isim kontrolü
            if sq.question_type == QuestionType.WHO:
                # Cevap "bilgi bulunamadı" içeriyorsa cevaplanmamış
                if "bulunamadı" in answer_lower or "bilinmiyor" in answer_lower:
                    answered = False
                # Cevap bir isim içeriyor mu?
                elif re.search(r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b', answer):
                    answered = True
                # Web sonuçlarından isim var mı?
                elif "Eş Genel Başkan" in answer and re.search(r'(Bakırhan|Tuncer|Tülay)', answer):
                    answered = True

            # HOW tipi sorular için süreç/yöntem kontrolü
            elif sq.question_type == QuestionType.HOW:
                # Süreç açıklaması var mı?
                process_keywords = ['seçilir', 'yapılır', 'belirlenir', 'oluşur', 'şekilde', 'yöntem']
                answered = any(kw in answer_lower for kw in process_keywords)

            # Diğer tipler için keyword eşleştirmesi
            else:
                keyword_matches = sum(1 for kw in sq.keywords if kw in answer_lower)
                answered = keyword_matches >= len(sq.keywords) * 0.3  # %30 eşleşme yeterli

            sq.answered = answered
            if not answered:
                unanswered.append(sq)

        is_complete = len(unanswered) == 0

        if not is_complete:
            logger.info(f"CompletenessCheck: {len(unanswered)} soru cevaplanmadı")
            for sq in unanswered:
                logger.info(f"  - [{sq.question_type.value}] {sq.text[:50]}...")

        return is_complete, unanswered

    def get_retrieval_strategy(self, sub_question: SubQuestion) -> Dict:
        """
        Alt soru için retrieval stratejisi belirler.

        Args:
            sub_question: Alt soru

        Returns:
            Dict: Retrieval parametreleri
        """
        strategy = {
            "use_web": sub_question.requires_web,
            "top_k": 3,
            "boost_keywords": [],
        }

        # WHO tipi - güncel isim gerektirir
        if sub_question.question_type == QuestionType.WHO:
            strategy["use_web"] = True
            strategy["top_k"] = 5

        # HOW tipi - tüzük bilgisi gerektirir
        elif sub_question.question_type == QuestionType.HOW:
            strategy["use_web"] = False
            strategy["top_k"] = 5
            strategy["boost_keywords"] = ["madde", "seçim", "yöntem"]

        return strategy


# Singleton
_analyzer_instance = None


def get_query_analyzer() -> QueryAnalyzer:
    """Singleton Query Analyzer döndürür."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = QueryAnalyzer()
    return _analyzer_instance
