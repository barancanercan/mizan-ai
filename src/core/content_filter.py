"""
mizan-ai - Content Filter
Küfür/Hakaret içeren sorguları filtreler
"""

import re
from typing import List, Tuple

OFFENSIVE_WORDS = [
    "orospu", "orospusu", "orospu çocuğu", "orospu çocuk", "orosbu",
    "piç", "pıç", "pic", "pıc", "pich", "pıch",
    "siktir", "siktirgit", "siktirgit", "sik", "sikim", "sikeyim",
    "bok", "bokum", "bokumun", "boktan",
    "göt", "got", "götveren", "gotveren", "götveren",
    "am", "amcık", "amcik", "amcığ", "amcığım", "amciğim",
    "sürtük", "surtuk", "surtuk",
    "şerefsiz", "serefsiz", "şerefsizim", "serefsizim",
    "hıyar", "hiyar", "hayın", "hain",
    "şapşal", "sapşal", "salak", "gerizekalı", "gerizekali",
    "aptal", "salak", "kalın", "kaltak",
    "fahişe", "fahise", "pespaye", "pespay",
    "muzır", "muzir", "adi", "adi",
    "şehvet", "sehuet", "fuhuş", "fuhus",
]

OFFENSIVE_PATTERNS = [
    r"\b(anani|ananı|ananın|ananın amcığı)\b",
    r"\b(babanı|babanın|babana)\b.*\b(siktir|sikeyim)\b",
    r"\b(seni|sene)\b.*\b(sikeyim|siktir)\b",
    r"\b(göt|got)\b.*\b(sokmak|sokar)\b",
    r"\b(am|amcık)\b.*\b(olmak|edecek)\b",
    r"\b(orospu|piç)\b.*\b(çocuk|çocuğu|babası)\b",
]


def is_offensive(query: str) -> Tuple[bool, str]:
    """
    Sorgunun hakaret/küfür içerip içermediğini kontrol eder.
    
    Args:
        query: Kullanıcı sorgusu
        
    Returns:
        Tuple[bool, str]: (hakaret_var, sebep)
    """
    query_lower = query.lower()
    
    for word in OFFENSIVE_WORDS:
        if word in query_lower:
            return True, f"Hakaret içeren kelime tespit edildi: {word}"
    
    for pattern in OFFENSIVE_PATTERNS:
        if re.search(pattern, query_lower):
            return True, "Hakaret içeren kalıp tespit edildi"
    
    return False, ""


def should_answer(query: str) -> Tuple[bool, str]:
    """
    Sorguya cevap verilip verilmeyeceğini belirler.
    
    Args:
        query: Kullanıcı sorgusu
        
    Returns:
        Tuple[bool, str]: (cevap_verilsin, mesaj)
    """
    is_bad, reason = is_offensive(query)
    
    if is_bad:
        return False, f"Bu soru uygunsuz içerik barındırdığı için yanıtlanamıyor. Siyasi konularda size yardımcı olmaktan memnuniyet duyarım. Saygılarımla."
    
    return True, ""


def filter_query(query: str) -> str:
    """
    Sorguyu temizler (opsiyonel).
    
    Args:
        query: Kullanıcı sorgusu
        
    Returns:
        str: Temizlenmiş sorgu
    """
    return query.strip()
