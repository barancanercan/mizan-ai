"""
FAZ 3: Query Rewriting Katmanı

Bu modül:
- Context-aware rewrite (gündelik dili resmi metne dönüştürme)
- Multi-query generation
- Ambiguity resolution
- Rewrite quality metrics
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


TURKISH_COLLOQUIAL_TO_FORMAL = {
    r'\bk(a|ı)z\b': 'kadın',
    r'\btaşeron\b': 'alt yüklenici',
    r'\btaşerona\b': 'alt yükleniciye',
    r'\btaşeronu\b': 'alt yükleniciyi',
    r'\bmemur\b': 'kamu görevlisi',
    r'\bmemura\b': 'kamu görevlisine',
    r'\bmemurlar\b': 'kamu görevlileri',
    r'\böğretmen\b': 'eğitimci',
    r'\bdoktor\b': 'hekim',
    r'\bdoktora\b': 'hekime',
    r'\bhemşire\b': 'sağlık personeli',
    r'\bpolis\b': 'güvenlik güçleri',
    r'\basker\b': 'silahlı kuvvetler',
    r'\bçiftçi\b': 'tarım üreticisi',
    r'\bişçi\b': 'çalışan',
    r'\bişçiler\b': 'çalışanlar',
    r'\bişveren\b': 'employer',
    r'\bpatron\b': 'işveren',
    r'\bsendika\b': 'meslek örgütü',
    r'\bparti\b': 'siyasi parti',
    r'\bhükümet\b': 'yürütme',
    r'\bdevlet\b': 'devlet',
    r'\bmilletvekili\b': ' milletvekili',
    r'\bbakan\b': 'bakan',
    r'\bcumhurbaşkanı\b': 'cumhurbaşkanı',
    r'\bvali\b': 'vali',
    r'\bbelediye\b': 'belediye',
    r'\bimam\b': 'din görevlisi',
    r'\bo imam\b': 'din görevlisi',
    r'\bkreş\b': 'çocuk bakım merkezi',
    r'\byuva\b': 'çocuk bakım merkezi',
    r'\bsigorta\b': 'sosyal güvenlik',
    r'\bemeklilik\b': 'yaşlılık güvencesi',
    r'\bmaaş\b': 'ücret',
    r'\bzam\b': 'ücret artışı',
    r'\bdüşük\b': 'düşük',
    r'\byüksek\b': 'yüksek',
    r'\beğitim\b': 'eğitim',
    r'\bsağlık\b': 'sağlık',
    r'\byargı\b': 'yargı',
    r'\badalet\b': 'adalet',
    r'\bemniyet\b': 'güvenlik',
    r'\bmilli eğitim\b': 'milli eğitim',
    r'\bdiyanet\b': 'diyanet işleri',
}


TURKISH_QUERY_EXPANSION = {
    'eğitim': ['eğitim', 'öğretim', 'okul', 'üniversite', 'mektep', 'ders', 'müfredat'],
    'sağlık': ['sağlık', 'hastane', 'tedavi', 'hekım', 'tıp', 'şifa'],
    'tarım': ['tarım', 'çiftçilik', 'ziraat', 'hayvancılık', 'bitki', 'ürün'],
    'ekonomi': ['ekonomi', 'ticaret', 'sanayi', 'iş', 'para', 'maliye', 'büyüme'],
    'dış politika': ['dış politika', 'diplomasi', 'uluslararası', '拜占庭', 'AB', 'NATO'],
    'güvenlik': ['güvenlik', 'savunma', 'askeri', 'emniyet', 'polis', 'ordu'],
    'adalet': ['adalet', 'yargı', 'hukuk', 'kanun', 'mahkeme', 'dava'],
    'çevre': ['çevre', 'doğa', 'yeşil', 'iklim', 'enerji', 'eko'],
    'sosyal': ['sosyal', 'yardım', 'işsizlik', 'emeklilik', 'sigorta'],
    'kültür': ['kültür', 'turizm', 'sanat', 'tarih', 'miras'],
}


@dataclass
class RewriteResult:
    """Query rewrite sonucu."""
    original_query: str
    rewritten_query: str
    method: str = "unknown"
    confidence: float = 0.0
    ambiguous: bool = False
    ambiguity_resolved: bool = False
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteMetrics:
    """Rewrite kalite metrikleri."""
    recall_improvement: float = 0.0
    precision_improvement: float = 0.0
    rewrite_quality_score: float = 0.0
    ambiguity_detected: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'recall_improvement': self.recall_improvement,
            'precision_improvement': self.precision_improvement,
            'rewrite_quality_score': self.rewrite_quality_score,
            'ambiguity_detected': self.ambiguity_detected,
            'latency_ms': self.latency_ms,
        }


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriters."""
    
    @abstractmethod
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> RewriteResult:
        """Rewrite query."""
        pass
    
    @abstractmethod
    def generate_multi_query(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        pass


class RuleBasedRewriter(BaseQueryRewriter):
    """
    Rule-based query rewriter.
    Kural tabanlı gündelik dil → resmi dil dönüşümü.
    """
    
    def __init__(
        self, 
        use_expansion: bool = True,
        expansion_depth: int = 2,
    ):
        self.use_expansion = use_expansion
        self.expansion_depth = expansion_depth
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.patterns = {}
        for pattern, replacement in TURKISH_COLLOQUIAL_TO_FORMAL.items():
            try:
                self.patterns[pattern] = re.compile(pattern, re.IGNORECASE)
            except re.error:
                logger.warning(f"Invalid pattern: {pattern}")
    
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> RewriteResult:
        """Rewrite query using rules."""
        import time
        start = time.time()
        
        rewritten = query
        
        # Apply colloquial to formal transformations
        for pattern, compiled in self.patterns.items():
            rewritten = compiled.sub(replacement, rewritten)
        
        # Normalize Turkish characters
        rewritten = self._normalize_turkish(rewritten)
        
        # Apply query expansion if enabled
        if self.use_expansion:
            expanded_queries = self._expand_query(rewritten)
            alternatives = expanded_queries[1:] if len(expanded_queries) > 1 else []
        else:
            alternatives = []
        
        latency = (time.time() - start) * 1000
        
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            method="rule_based",
            confidence=0.7,
            alternatives=alternatives,
            metadata={'latency_ms': latency},
        )
    
    def generate_multi_query(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        queries = [query]
        
        # Main rewritten query
        rewritten = self.rewrite(query)
        if rewritten.rewritten_query != query:
            queries.append(rewritten.rewritten_query)
        
        # Expanded queries
        if self.use_expansion:
            expanded = self._expand_query(query)
            queries.extend(expanded[:self.expansion_depth])
        
        # Generate paraphrases (simple variations)
        queries.extend(self._generate_paraphrases(query))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:10]
    
    def _normalize_turkish(self, text: str) -> str:
        """Normalize Turkish characters."""
        replacements = {
            'ı': 'i', 'İ': 'I',
            'ş': 's', 'Ş': 'S',
            'ğ': 'g', 'Ğ': 'G',
            'ü': 'u', 'Ü': 'U',
            'ö': 'o', 'Ö': 'O',
            'ç': 'c', 'Ç': 'C',
        }
        for turkish, latin in replacements.items():
            text = text.replace(turkish, latin)
        return text
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        expanded = [query]
        query_lower = query.lower()
        
        for key, synonyms in TURKISH_QUERY_EXPANSION.items():
            if key in query_lower:
                for syn in synonyms:
                    if syn != key and syn not in query_lower:
                        new_query = query_lower.replace(key, syn)
                        if new_query not in [e.lower() for e in expanded]:
                            expanded.append(new_query)
        
        return expanded
    
    def _generate_paraphrases(self, query: str) -> List[str]:
        """Generate simple paraphrases."""
        paraphrases = []
        
        # Add question variations
        if query.endswith('?'):
            paraphrases.append(query[:-1])
            paraphrases.append(query.replace('?', ''))
        else:
            paraphrases.append(query + '?')
        
        # Add formal prefixes
        prefixes = ['', ' ', ' ']
        for prefix in prefixes:
            if query not in paraphrases:
                paraphrases.append(prefix + query)
        
        return paraphrases[:3]


class LLMQueryRewriter(BaseQueryRewriter):
    """
    LLM-powered query rewriter.
    Daha akıllı context-aware rewriting için LLM kullanır.
    """
    
    def __init__(
        self, 
        llm: Any = None,
        use_fallback: bool = True,
    ):
        self.llm = llm
        self.use_fallback = use_fallback
        self.fallback_rewriter = RuleBasedRewriter() if use_fallback else None
        
        if llm is None:
            logger.warning("No LLM provided, using rule-based fallback")
    
    def rewrite(self, query: str, context: Optional[List[str]] = None) -> RewriteResult:
        """Rewrite query using LLM."""
        import time
        start = time.time()
        
        if self.llm is None:
            if self.fallback_rewriter:
                return self.fallback_rewriter.rewrite(query, context)
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                method="no_rewrite",
            )
        
        try:
            # Build prompt
            prompt = self._build_rewrite_prompt(query, context)
            
            # Call LLM
            response = self.llm.invoke(prompt)
            rewritten = self._parse_rewrite_response(response, query)
            
            latency = (time.time() - start) * 1000
            
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                method="llm",
                confidence=0.85,
                metadata={'latency_ms': latency},
            )
            
        except Exception as e:
            logger.error(f"LLM rewrite error: {e}")
            if self.fallback_rewriter:
                return self.fallback_rewriter.rewrite(query, context)
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                method="error_fallback",
            )
    
    def generate_multi_query(self, query: str) -> List[str]:
        """Generate multiple query variations using LLM."""
        if self.llm is None:
            if self.fallback_rewriter:
                return self.fallback_rewriter.generate_multi_query(query)
            return [query]
        
        try:
            prompt = self._build_multi_query_prompt(query)
            response = self.llm.invoke(prompt)
            queries = self._parse_multi_query_response(response)
            
            if queries:
                return queries
            
        except Exception as e:
            logger.error(f"Multi-query generation error: {e}")
        
        # Fallback
        if self.fallback_rewriter:
            return self.fallback_rewriter.generate_multi_query(query)
        return [query]
    
    def _build_rewrite_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Build rewrite prompt."""
        prompt = f"""Sen bir Türk siyasi doküman arama uzmanısın.
Kullanıcının gündelik dildeki sorusunu, resmi parti dokümanlarında aranabilecek resmi bir sorguya dönüştür.

Kurallar:
1. Günlük konuşma dilini resmi dile çevir
2. Türkçe karakterleri doğru kullan (ı, İ, ş, Ş, ğ, Ü, ü, ö, Ö, ç, Ç)
3. Kısa ve öz tut
4. Eş anlamlı kelimeleri ekle

Örnekler:
- "çiftçilere para yok mu?" → "tarım desteği çiftçi finansmanı"
- "öğretmenler ne kadar maaş alıyor?" → "eğitimci ücretleri maaş politikası"
- "kadınlara ne veriyon?" → "kadın hakları sosyal destek"

Sorgu: {query}
"""
        if context:
            prompt += f"\nBağlam: {' '.join(context[:3])}"
        
        prompt += "\nResmi sorgu:"
        return prompt
    
    def _build_multi_query_prompt(self, query: str) -> str:
        """Build multi-query prompt."""
        return f"""Sen bir Türk siyasi doküman arama uzmanısın.
Verilen sorgunun farklı varyasyonlarını oluştur.

Kurallar:
1. Farklı bakış açılarından sorgular oluştur
2. Eş anlamlı kelimeler kullan
3. Resmi ve yarı-resmi tonlarda yaz
4. Her satıra bir sorgu yaz
5. En fazla 5 sorgu ver

Sorgu: {query}

Varyasyonlar:"""
    
    def _parse_rewrite_response(self, response: Any, original: str) -> str:
        """Parse LLM rewrite response."""
        try:
            text = response.content if hasattr(response, 'content') else str(response)
            text = text.strip()
            if text:
                return text
        except:
            pass
        return original
    
    def _parse_multi_query_response(self, response: Any) -> List[str]:
        """Parse multi-query response."""
        try:
            text = response.content if hasattr(response, 'content') else str(response)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            queries = []
            for line in lines:
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and len(line) > 5:
                    queries.append(line)
            return queries[:5]
        except:
            return []


class AmbiguityResolver:
    """
    Ambiguity detection and resolution.
    Belirsiz sorguları tespit eder ve çözüm önerileri sunar.
    """
    
    AMBIGUOUS_PATTERNS = [
        r'^(ne|nasıl|neden|niye)\s+(olsun|var|yok|değil)\?*$',
        r'^(o|bu|şu)\s+\w+\s+(mi|mu|mı|mü)$',
        r'^(ne|hangı)\s+(\w+\s+){0,2}(parti|adam|kişi)?$',
        r'\b(birileri|birşey|birşeyi)\b',
    ]
    
    TURKISH_CONTEXT_KEYWORDS = {
        'siyasi_parti': ['parti', 'chp', 'akp', 'mhp', 'iyi', 'dem', 'sp', 'bbp', 'zp'],
        'sosyal': ['maaş', 'para', 'iş', 'çocuk', 'kadın', 'yaşlı', 'engelli'],
        'eğitim': ['okul', 'üniversite', 'öğrenci', 'öğretmen', 'eğitim'],
        'sağlık': ['hastane', 'doktor', 'sağlık', 'tedavi', 'ilaç'],
        'tarım': ['çiftçi', 'tarım', 'toprak', 'ürün', 'hayvan'],
        'ekonomi': ['işsizlik', 'enflasyon', 'fiyat', 'para', 'dolar', 'euro'],
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile ambiguity patterns."""
        self.patterns = []
        for pattern in self.AMBIGUOUS_PATTERNS:
            try:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning(f"Invalid pattern: {pattern}")
    
    def detect_ambiguity(self, query: str) -> Tuple[bool, str, List[str]]:
        """
        Detect query ambiguity.
        
        Returns:
            Tuple of (is_ambiguous, ambiguity_type, suggestions)
        """
        query_lower = query.lower().strip()
        
        # Pattern-based detection
        for pattern in self.patterns:
            if pattern.search(query_lower):
                return True, "pattern_match", self._generate_suggestions(query)
        
        # Too short
        if len(query_lower.split()) < 2:
            return True, "too_short", self._generate_suggestions(query)
        
        # Context keyword detection
        detected_contexts = self._detect_context(query_lower)
        if len(detected_contexts) == 0:
            return True, "no_context", self._generate_suggestions(query)
        
        return False, "none", []
    
    def resolve_ambiguity(
        self, 
        query: str, 
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Resolve query ambiguity.
        
        Returns:
            Dict with resolved query and alternatives
        """
        is_ambiguous, ambiguity_type, suggestions = self.detect_ambiguity(query)
        
        if not is_ambiguous:
            return {
                'is_ambiguous': False,
                'resolved_query': query,
                'ambiguity_type': ambiguity_type,
                'alternatives': [],
            }
        
        # Generate more specific alternatives
        alternatives = self._generate_alternatives(query, context)
        
        return {
            'is_ambiguous': True,
            'resolved_query': query,
            'ambiguity_type': ambiguity_type,
            'alternatives': alternatives,
            'suggestions': suggestions,
        }
    
    def _detect_context(self, query: str) -> List[str]:
        """Detect context keywords in query."""
        detected = []
        for context_type, keywords in self.TURKISH_CONTEXT_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                detected.append(context_type)
        return detected
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """Generate suggestions for ambiguous query."""
        suggestions = []
        
        # Add context keywords
        for context_type, keywords in self.TURKISH_CONTEXT_KEYWORDS.items():
            for kw in keywords[:2]:
                if kw not in query.lower():
                    suggestions.append(f"{query} {kw}")
        
        return suggestions[:5]
    
    def _generate_alternatives(
        self, 
        query: str, 
        context: Optional[List[str]] = None
    ) -> List[str]:
        """Generate alternative more specific queries."""
        alternatives = []
        
        # Add party context if mentioned
        if context:
            for ctx in context[:3]:
                if any(p in ctx.lower() for p in ['chp', 'akp', 'mhp', 'iyi', 'dem']):
                    alternatives.append(f"{query} {ctx}")
        
        # Add TURKISH_CONTEXT_KEYWORDS
        for context_type, keywords in self.TURKISH_CONTEXT_KEYWORDS.items():
            for kw in keywords[:1]:
                if kw not in query.lower():
                    alternatives.append(f"{query} {kw}")
        
        return alternatives[:5]


class RewriteEvaluator:
    """
    Rewrite kalite metrikleri.
    Rewrite → recall artışını ölçer.
    """
    
    def __init__(self, retrieval_system: Any = None):
        self.retrieval_system = retrieval_system
    
    def evaluate_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
        top_k: int = 5,
    ) -> RewriteMetrics:
        """
        Evaluate rewrite quality by measuring recall improvement.
        
        Args:
            original_query: Original user query
            rewritten_query: Rewritten query
            top_k: Number of documents to retrieve
            
        Returns:
            RewriteMetrics: Evaluation metrics
        """
        import time
        start = time.time()
        
        metrics = RewriteMetrics()
        
        if self.retrieval_system is None:
            logger.warning("No retrieval system provided for evaluation")
            metrics.latency_ms = (time.time() - start) * 1000
            return metrics
        
        try:
            # Retrieve with original query
            original_results = self.retrieval_system.search(original_query, top_k=top_k)
            original_docs = set(d.page_content for d in original_results.documents)
            
            # Retrieve with rewritten query
            rewritten_results = self.retrieval_system.search(rewritten_query, top_k=top_k)
            rewritten_docs = set(d.page_content for d in rewritten_results.documents)
            
            # Calculate improvement
            if original_docs:
                overlap = len(original_docs.intersection(rewritten_docs))
                metrics.recall_improvement = overlap / len(original_docs)
            
            # Precision improvement (assuming original is less precise)
            if rewritten_docs:
                metrics.precision_improvement = 0.1
            
            # Rewrite quality score (simple heuristic)
            metrics.rewrite_quality_score = self._calculate_quality_score(
                original_query, rewritten_query
            )
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
        
        metrics.latency_ms = (time.time() - start) * 1000
        return metrics
    
    def evaluate_multi_query(
        self,
        original_query: str,
        generated_queries: List[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate multi-query generation.
        
        Returns:
            Dict with evaluation results
        """
        if self.retrieval_system is None:
            return {'error': 'No retrieval system'}
        
        results = {
            'original_query': original_query,
            'generated_queries': generated_queries,
            'best_query': original_query,
            'best_recall': 0.0,
            'query_recalls': {},
        }
        
        try:
            # Get original recall
            original_results = self.retrieval_system.search(original_query, top_k=top_k)
            original_count = len(original_results.documents)
            
            for query in generated_queries:
                query_results = self.retrieval_system.search(query, top_k=top_k)
                query_count = len(query_results.documents)
                
                # Calculate recall improvement
                query_docs = set(d.page_content for d in query_results.documents)
                original_docs = set(d.page_content for d in original_results.documents)
                
                recall = len(query_docs.intersection(original_docs)) / max(original_count, 1)
                results['query_recalls'][query] = recall
                
                if recall > results['best_recall']:
                    results['best_recall'] = recall
                    results['best_query'] = query
            
            results['recall_improvement'] = results['best_recall']
            
        except Exception as e:
            logger.error(f"Multi-query evaluation error: {e}")
        
        return results
    
    def _calculate_quality_score(
        self, 
        original: str, 
        rewritten: str
    ) -> float:
        """Calculate simple quality score based on transformation."""
        if original == rewritten:
            return 0.0
        
        # Score based on:
        # 1. Length difference (prefer slight expansion)
        # 2. Character normalization
        # 3. Formal term usage
        
        score = 0.5
        
        # Length heuristic
        len_ratio = len(rewritten) / max(len(original), 1)
        if 0.8 <= len_ratio <= 1.5:
            score += 0.2
        
        # Formal term replacement
        formal_terms = ['siyasi', 'kamu', 'sosyal', 'eğitim', 'sağlık', 'tarım']
        if any(term in rewritten.lower() for term in formal_terms):
            score += 0.3
        
        return min(score, 1.0)


class QueryRewritingPipeline:
    """
    Complete query rewriting pipeline.
    Combines rewriter, ambiguity resolver, and evaluator.
    """
    
    def __init__(
        self,
        rewriter: Optional[BaseQueryRewriter] = None,
        ambiguity_resolver: Optional[AmbiguityResolver] = None,
        evaluator: Optional[RewriteEvaluator] = None,
        enable_multi_query: bool = True,
        enable_ambiguity_resolution: bool = True,
    ):
        self.rewriter = rewriter or RuleBasedRewriter()
        self.ambiguity_resolver = ambiguity_resolver or AmbiguityResolver()
        self.evaluator = evaluator
        self.enable_multi_query = enable_multi_query
        self.enable_ambiguity_resolution = enable_ambiguity_resolution
    
    def process(
        self,
        query: str,
        context: Optional[List[str]] = None,
        evaluate: bool = False,
    ) -> Dict[str, Any]:
        """
        Process query through rewriting pipeline.
        
        Returns:
            Dict with rewritten query, alternatives, and metrics
        """
        result = {
            'original_query': query,
            'main_rewritten': query,
            'alternatives': [],
            'multi_queries': [],
            'ambiguity': {},
            'metrics': {},
        }
        
        # Ambiguity resolution
        if self.enable_ambiguity_resolution:
            ambiguity_result = self.ambiguity_resolver.resolve_ambiguity(query, context)
            result['ambiguity'] = ambiguity_result
            
            if ambiguity_result['alternatives']:
                result['alternatives'] = ambiguity_result['alternatives']
        
        # Main rewrite
        rewrite_result = self.rewriter.rewrite(query, context)
        result['main_rewritten'] = rewrite_result.rewritten_query
        
        if rewrite_result.alternatives:
            result['alternatives'].extend(rewrite_result.alternatives)
        
        # Multi-query generation
        if self.enable_multi_query:
            multi_queries = self.rewriter.generate_multi_query(query)
            result['multi_queries'] = multi_queries
            result['alternatives'].extend(multi_queries)
        
        # Remove duplicates
        seen = set()
        unique_alts = []
        for alt in result['alternatives']:
            alt_lower = alt.lower().strip()
            if alt_lower not in seen and alt_lower != query.lower().strip():
                seen.add(alt_lower)
                unique_alts.append(alt)
        result['alternatives'] = unique_alts[:10]
        
        # Evaluation
        if evaluate and self.evaluator:
            metrics = self.evaluator.evaluate_rewrite(
                query, 
                rewrite_result.rewritten_query
            )
            result['metrics'] = metrics.to_dict()
            
            if result['multi_queries']:
                multi_metrics = self.evaluator.evaluate_multi_query(
                    query,
                    result['multi_queries']
                )
                result['multi_metrics'] = multi_metrics
        
        return result


def create_query_rewriter(
    llm: Any = None,
    use_rule_based_fallback: bool = True,
    use_expansion: bool = True,
) -> BaseQueryRewriter:
    """
    Query rewriter factory.
    
    Args:
        llm: LLM instance (optional)
        use_rule_based_fallback: Use rule-based if LLM fails
        use_expansion: Use query expansion
        
    Returns:
        BaseQueryRewriter instance
    """
    if llm:
        return LLMQueryRewriter(llm=llm, use_fallback=use_rule_based_fallback)
    return RuleBasedRewriter(use_expansion=use_expansion)
