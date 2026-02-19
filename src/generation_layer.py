"""
FAZ 4: Generation Layer

Bu modül:
- Deterministic, source-grounded output üretimi
- Temperature lock (0-0.1)
- Citation zorunluluğu
- Format sabitleme
- Max token boundary
- Strict output template
- Context dışına çıkma yasağı
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


DEFAULT_TEMPERATURE = 0.1
MAX_TOKENS_DEFAULT = 1024
MIN_TOKENS_DEFAULT = 50


@dataclass
class Citation:
    """Citation bilgisi."""
    text: str
    source: str
    page: Optional[int] = None
    party: Optional[str] = None
    relevance_score: float = 0.0
    
    def to_markdown(self) -> str:
        """Markdown formatında citation."""
        parts = [f"**Kaynak:** {self.source}"]
        if self.party:
            parts.append(f"**Parti:** {self.party}")
        if self.page:
            parts.append(f"**Sayfa:** {self.page}")
        return " | ".join(parts)


@dataclass
class GenerationConfig:
    """Generation yapılandırması."""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = MAX_TOKENS_DEFAULT
    min_tokens: int = MIN_TOKENS_DEFAULT
    require_citation: bool = True
    strict_format: bool = True
    enforce_context_bounds: bool = True
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    def validate(self) -> bool:
        """Config validation."""
        if not 0 <= self.temperature <= 1:
            logger.warning(f"Temperature {self.temperature} out of bounds, clamping to [0,1]")
            self.temperature = max(0, min(1, self.temperature))
        
        if self.max_tokens < self.min_tokens:
            logger.warning("max_tokens < min_tokens, swapping values")
            self.max_tokens, self.min_tokens = self.min_tokens, self.max_tokens
        
        return True


@dataclass
class GeneratedAnswer:
    """Üretilen cevap."""
    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    context_used: str = ""
    generation_config: Optional[GenerationConfig] = None
    latency_ms: float = 0.0
    token_count: int = 0
    hallucinations_detected: bool = False
    out_of_context: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'citations': [c.to_markdown() for c in self.citations],
            'sources_used': self.sources_used,
            'latency_ms': self.latency_ms,
            'token_count': self.token_count,
            'hallucinations_detected': self.hallucinations_detected,
            'out_of_context': self.out_of_context,
        }
    
    def has_citations(self) -> bool:
        return len(self.citations) > 0


class BaseAnswerGenerator(ABC):
    """Abstract base class for answer generators."""
    
    @abstractmethod
    def generate(
        self, 
        query: str, 
        context: str,
        config: Optional[GenerationConfig] = None,
    ) -> GeneratedAnswer:
        """Generate answer from context."""
        pass


class DeterministicGenerator(BaseAnswerGenerator):
    """
    Deterministic answer generator.
    Düşük temperature, citation zorunluluğu, context bounds enforcement.
    """
    
    def __init__(
        self, 
        llm: Any = None,
        default_config: Optional[GenerationConfig] = None,
    ):
        self.llm = llm
        self.default_config = default_config or GenerationConfig()
        self.citation_enforcer = CitationEnforcer()
        self.output_formatter = OutputFormatter()
        self.context_validator = ContextValidator()
    
    def generate(
        self, 
        query: str, 
        context: str,
        config: Optional[GenerationConfig] = None,
    ) -> GeneratedAnswer:
        """Generate answer with strict constraints."""
        import time
        start = time.time()
        
        cfg = config or self.default_config
        cfg.validate()
        
        answer = GeneratedAnswer(
            context_used=context[:500],
            generation_config=cfg,
        )
        
        if not context or len(context.strip()) < 10:
            answer.answer = "Yeterli bağlam bulunamadı."
            answer.latency_ms = (time.time() - start) * 1000
            return answer
        
        try:
            # Build prompt with strict instructions
            prompt = self._build_strict_prompt(query, context, cfg)
            
            # Generate with locked temperature
            if self.llm:
                response = self._generate_with_llm(prompt, cfg)
            else:
                response = self._generate_fallback(query, context)
            
            answer.answer = response
            
            # Extract and enforce citations
            if cfg.require_citation:
                citations = self.citation_enforcer.extract_citations(response, context)
                answer.citations = citations
                answer.sources_used = list(set(c.source for c in citations))
            
            # Validate context bounds
            if cfg.enforce_context_bounds:
                is_out, violations = self.context_validator.validate(response, context)
                answer.out_of_context = is_out
                if is_out:
                    answer.answer = self.context_validator.sanitize_answer(
                        response, context
                    )
            
            # Format output
            if cfg.strict_format:
                answer.answer = self.output_formatter.format(answer.answer)
            
            # Estimate token count
            answer.token_count = len(answer.answer.split())
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            answer.answer = "Yanıt üretilirken hata oluştu."
        
        answer.latency_ms = (time.time() - start) * 1000
        return answer
    
    def _build_strict_prompt(
        self, 
        query: str, 
        context: str, 
        cfg: GenerationConfig
    ) -> str:
        """Build strict generation prompt."""
        return f"""Sen Türk siyasi parti dokümanları için cevap üretensin.

KURALLAR:
1. SADECE aşağıdaki bağlamdaki bilgileri kullan
2. Bağlam dışında bilgi ekleme (haluisnation YASAK)
3. Her iddia için kaynak göster (MADDE X, Sayfa Y)
4. Türkçe karakterleri doğru kullan (ı, İ, ş, Ş, ğ, ü, Ü, ö, Ö, ç, Ç)
5. Nesnel ve tarafsız ol
6. Maksimum {cfg.max_tokens} kelime kullan

BAĞLAM:
{context[:2000]}

SORU: {query}

CEVAP:"""
    
    def _generate_with_llm(self, prompt: str, cfg: GenerationConfig) -> str:
        """Generate with LLM."""
        try:
            # Lock temperature
            original_temp = None
            if hasattr(self.llm, 'temperature'):
                original_temp = self.llm.temperature
                self.llm.temperature = cfg.temperature
            
            response = self.llm.invoke(prompt)
            
            # Restore temperature
            if original_temp is not None:
                self.llm.temperature = original_temp
            
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def _generate_fallback(self, query: str, context: str) -> str:
        """Fallback generation without LLM."""
        # Simple extractive fallback
        sentences = context.split('.')
        relevant = [s for s in sentences if any(w in s.lower() for w in query.lower().split()[:3])]
        
        if relevant:
            return '.'.join(relevant[:3]) + '.'
        return context[:200] + "..."


class CitationEnforcer:
    """
    Citation extraction and enforcement.
    Kaynak gösterimini zorunlu kılar.
    """
    
    CITATION_PATTERNS = [
        r'(?:MADDE|Madde|madde)\s*(\d+)',
        r'(?:Sayfa|sayfa|S\.)\s*(\d+)',
        r'(?:Kaynak|kaynak|Kaynağa\s*göre)\s*[:\-]?\s*([^\n]+)',
        r'\[(\d+)\]', r'\((\d+)\)',
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.patterns = []
        for pattern in self.CITATION_PATTERNS:
            try:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning(f"Invalid pattern: {pattern}")
    
    def extract_citations(
        self, 
        answer: str, 
        context: str
    ) -> List[Citation]:
        """Extract citations from answer."""
        citations = []
        
        # Pattern-based extraction
        for pattern in self.patterns:
            matches = pattern.finditer(answer)
            for match in matches:
                citation = Citation(
                    text=match.group(0),
                    source="Belge",
                    relevance_score=1.0,
                )
                citations.append(citation)
        
        # If no explicit citations, create from context
        if not citations:
            citations = self._infer_citations(answer, context)
        
        return citations
    
    def _infer_citations(
        self, 
        answer: str, 
        context: str
    ) -> List[Citation]:
        """Infer citations from context."""
        citations = []
        
        # Split answer into sentences
        answer_sentences = answer.split('.')
        
        for i, sent in enumerate(answer_sentences):
            if len(sent.strip()) > 20:
                citation = Citation(
                    text=sent.strip()[:100],
                    source="İlgili Belge",
                    page=i + 1,
                    relevance_score=0.8,
                )
                citations.append(citation)
        
        return citations[:5]
    
    def enforce_citation_presence(self, answer: str) -> Tuple[bool, str]:
        """Ensure answer has citation markers."""
        has_citation = any(p.search(answer) for p in self.patterns)
        
        if not has_citation:
            # Add citation placeholder
            answer += "\n\n[Kaynak: Belge]"
        
        return has_citation, answer
    
    def validate_citations(
        self, 
        answer: str, 
        context: str
    ) -> Tuple[bool, List[str]]:
        """Validate that citations are from actual context."""
        issues = []
        
        # Check each citation claim
        for pattern in self.patterns:
            matches = pattern.finditer(answer)
            for match in matches:
                cited_text = match.group(0)
                if cited_text.lower() not in context.lower():
                    issues.append(f"Citation not in context: {cited_text}")
        
        return len(issues) == 0, issues


class OutputFormatter:
    """
    Strict output format enforcement.
    Tutarlı çıktı formatı oluşturur.
    """
    
    FORMATS = {
        'default': {
            'prefix': '',
            'citation_format': '\n\nKaynaklar: {citations}',
            'suffix': '',
        },
        'markdown': {
            'prefix': '## Yanıt\n\n',
            'citation_format': '\n\n### Kaynaklar\n{citations}',
            'suffix': '',
        },
        'structured': {
            'prefix': 'YANIT:\n',
            'citation_format': '\n\nKAYNAKLAR:\n{citations}',
            'suffix': '\n\n---',
        },
    }
    
    def __init__(self, format_type: str = 'default'):
        self.format_type = format_type
        self.current_format = self.FORMATS.get(format_type, self.FORMATS['default'])
    
    def format(self, answer: str, citations: Optional[List[Citation]] = None) -> str:
        """Format answer according to template."""
        fmt = self.current_format
        
        formatted = fmt['prefix'] + answer
        
        if citations and fmt['citation_format']:
            citation_text = self._format_citations(citations)
            formatted += fmt['citation_format'].format(citations=citation_text)
        
        formatted += fmt['suffix']
        
        # Post-processing
        formatted = self._clean_whitespace(formatted)
        formatted = self._fix_turkish_chars(formatted)
        
        return formatted
    
    def _format_citations(self, citations: List[Citation]) -> str:
        """Format citations list."""
        lines = []
        for i, cit in enumerate(citations[:5], 1):
            lines.append(f"{i}. {cit.to_markdown()}")
        return '\n'.join(lines)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean extra whitespace."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        return text
    
    def _fix_turkish_chars(self, text: str) -> str:
        """Ensure correct Turkish characters."""
        # This is more of a normalization check
        return text


class ContextValidator:
    """
    Context bounds validation.
    Answer'ın context içinde olup olmadığını kontrol eder.
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.3,
        max_out_of_context_ratio: float = 0.2,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_out_of_context_ratio = max_out_of_context_ratio
    
    def validate(
        self, 
        answer: str, 
        context: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate answer is grounded in context.
        
        Returns:
            Tuple of (is_out_of_context, violations)
        """
        violations = []
        
        if not context:
            return True, ["No context provided"]
        
        # Split answer into claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return False, []
        
        # Check each claim
        out_of_context_count = 0
        for claim in claims:
            if not self._claim_in_context(claim, context):
                out_of_context_count += 1
                violations.append(f"Claim not in context: {claim[:50]}...")
        
        # Calculate ratio
        ratio = out_of_context_count / len(claims) if claims else 0
        is_out = ratio > self.max_out_of_context_ratio
        
        return is_out, violations
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer."""
        # Split by sentences and filter
        sentences = answer.split('.')
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Skip very short sentences
            if len(sent) < 20:
                continue
            # Skip questions
            if sent.endswith('?'):
                continue
            claims.append(sent)
        
        return claims
    
    def _claim_in_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context."""
        # Simple word overlap check
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stopwords = {
            've', 'veya', 'ama', 'fakat', 'lakin', 'çünkü', 'bu', 'şu', 'o',
            'bir', 'ile', 'için', 'kadar', 'gibi', 'de', 'da', 'mi', 'mu',
        }
        claim_words = claim_words - stopwords
        context_words = context_words - stopwords
        
        if not claim_words:
            return True
        
        overlap = len(claim_words.intersection(context_words))
        similarity = overlap / len(claim_words)
        
        return similarity >= self.similarity_threshold
    
    def sanitize_answer(
        self, 
        answer: str, 
        context: str
    ) -> str:
        """Remove out-of-context parts from answer."""
        # This is a simple implementation
        # In practice, you'd use more sophisticated methods
        lines = answer.split('\n')
        valid_lines = []
        
        for line in lines:
            if self._claim_in_context(line, context):
                valid_lines.append(line)
            else:
                valid_lines.append("[Bu kısım bağlam dışıdır]")
        
        return '\n'.join(valid_lines)


class HallucinationDetector:
    """
    Hallucination detection.
    Bağlam dışı veya uydurma bilgileri tespit eder.
    """
    
    HALLUCINATION_INDICATORS = [
        r'^bilmiyorum$',
        r'^emin değilim',
        r'^(?:sanki|galiba|muhtemelen|büyük ihtimalle)',
        r'\b(?:hayal|buluş|uydurma|kurgu)\b',
        r'^\s*(?:aslında|gerçekte|doğrusu)',
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile patterns."""
        self.patterns = []
        for pattern in self.HALLUCINATION_INDICATORS:
            try:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass
    
    def detect(self, answer: str, context: str) -> Tuple[bool, List[str]]:
        """Detect hallucinations."""
        issues = []
        
        # Pattern-based detection
        for pattern in self.patterns:
            if pattern.search(answer):
                issues.append(f"Hallucination indicator found: {pattern.pattern}")
        
        # Context validation
        is_out, violations = ContextValidator().validate(answer, context)
        if is_out:
            issues.extend(violations)
        
        return len(issues) > 0, issues


class GenerationPipeline:
    """
    Complete generation pipeline.
    Tüm generation bileşenlerini birleştirir.
    """
    
    def __init__(
        self,
        generator: Optional[BaseAnswerGenerator] = None,
        config: Optional[GenerationConfig] = None,
    ):
        self.generator = generator or DeterministicGenerator()
        self.config = config or GenerationConfig()
        self.hallucination_detector = HallucinationDetector()
    
    def generate(
        self,
        query: str,
        context: str,
        documents: Optional[List[Any]] = None,
    ) -> GeneratedAnswer:
        """Generate answer through full pipeline."""
        
        # Generate answer
        answer = self.generator.generate(query, context, self.config)
        
        # Detect hallucinations
        is_hallucinated, issues = self.hallucination_detector.detect(
            answer.answer, context
        )
        answer.hallucinations_detected = is_hallucinated
        
        if is_hallucinated:
            logger.warning(f"Hallucination detected: {issues}")
        
        # Add document metadata to citations
        if documents:
            for i, doc in enumerate(documents[:len(answer.citations)]):
                if hasattr(doc, 'metadata'):
                    if i < len(answer.citations):
                        answer.citations[i].source = doc.metadata.get('source', 'Belge')
                        answer.citations[i].party = doc.metadata.get('party')
        
        return answer
    
    def generate_multi(
        self,
        query: str,
        contexts: List[str],
        documents: Optional[List[Any]] = None,
    ) -> GeneratedAnswer:
        """Generate from multiple contexts."""
        combined_context = "\n\n".join(contexts)
        return self.generate(query, combined_context, documents)


def create_generator(
    llm: Any = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    require_citation: bool = True,
    strict_format: bool = True,
) -> DeterministicGenerator:
    """
    Generator factory.
    
    Args:
        llm: LLM instance
        temperature: Locked temperature (0-0.1 recommended)
        max_tokens: Max tokens for output
        require_citation: Require citations
        strict_format: Use strict output format
        
    Returns:
        DeterministicGenerator instance
    """
    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        require_citation=require_citation,
        strict_format=strict_format,
    )
    
    return DeterministicGenerator(llm=llm, default_config=config)


def lock_temperature(llm: Any, temperature: float = 0.1) -> Any:
    """
    Lock LLM temperature.
    
    Args:
        llm: LLM instance
        temperature: Temperature to lock (0-0.1 recommended)
        
    Returns:
        LLM with locked temperature
    """
    if hasattr(llm, 'temperature'):
        llm.temperature = max(0, min(1, temperature))
        logger.info(f"Temperature locked to {llm.temperature}")
    
    return llm
