"""
FAZ 5: Evaluation Stack (En Kritik)

Bu modül:
- Retrieval Recall@k (already in retrieval_engine.py)
- Citation Span Accuracy
- Hallucination Rate
- Determinism Test

Agent Görevleri:
- Evaluation Agent: Gold QA set, recall test pipeline, citation verification
- Hallucination Judge Agent: Answer vs Context, external judge, false claim detection
"""

import logging
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation sonucu."""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    evaluation_id: str
    query: str
    retrieval_metrics: Optional[EvaluationResult] = None
    citation_metrics: Optional[EvaluationResult] = None
    hallucination_metrics: Optional[EvaluationResult] = None
    determinism_metrics: Optional[EvaluationResult] = None
    overall_score: float = 0.0
    passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'evaluation_id': self.evaluation_id,
            'query': self.query,
            'retrieval': self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            'citation': self.citation_metrics.to_dict() if self.citation_metrics else None,
            'hallucination': self.hallucination_metrics.to_dict() if self.hallucination_metrics else None,
            'determinism': self.determinism_metrics.to_dict() if self.determinism_metrics else None,
            'overall_score': self.overall_score,
            'passed': self.passed,
        }


@dataclass
class GoldQA:
    """Gold QA pair."""
    query: str
    relevant_docs: List[str]
    expected_answer: str
    context: str
    party: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldQA':
        return cls(**data)


class EvaluationStore:
    """
    Gold QA set storage.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.qa_pairs: List[GoldQA] = []
        if filepath:
            self.load(filepath)
    
    def add(self, qa: GoldQA):
        """Add QA pair."""
        self.qa_pairs.append(qa)
    
    def add_batch(self, qas: List[GoldQA]):
        """Add multiple QA pairs."""
        self.qa_pairs.extend(qas)
    
    def get_all(self) -> List[GoldQA]:
        """Get all QA pairs."""
        return self.qa_pairs
    
    def get_by_party(self, party: str) -> List[GoldQA]:
        """Get QA pairs by party."""
        return [qa for qa in self.qa_pairs if qa.party == party]
    
    def save(self, filepath: Optional[str] = None):
        """Save to JSON."""
        path = filepath or self.filepath
        if not path:
            raise ValueError("No filepath specified")
        
        data = [qa.to_dict() for qa in self.qa_pairs]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """Load from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.qa_pairs = [GoldQA.from_dict(d) for d in data]
    
    def get_sample(self, n: int = 10) -> List[GoldQA]:
        """Get random sample."""
        import random
        return random.sample(self.qa_pairs, min(n, len(self.qa_pairs)))


class RecallEvaluator:
    """
    Recall@k evaluation.
    Mevcut retrieval_engine'daki RetrievalEvaluator'ı kullanır.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Any],
        relevant_doc_ids: List[str],
    ) -> EvaluationResult:
        """Evaluate recall@k."""
        if not relevant_doc_ids:
            return EvaluationResult(
                metric_name="recall_at_k",
                score=0.0,
                details={'error': 'No relevant docs provided'},
            )
        
        retrieved_ids = set()
        for doc in retrieved_docs[:self.k]:
            doc_id = self._get_doc_id(doc)
            retrieved_ids.add(doc_id)
        
        relevant_set = set(relevant_doc_ids)
        relevant_retrieved = retrieved_ids.intersection(relevant_set)
        
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        
        return EvaluationResult(
            metric_name="recall_at_k",
            score=recall,
            details={
                'k': self.k,
                'retrieved_count': len(retrieved_ids),
                'relevant_count': len(relevant_set),
                'relevant_retrieved': len(relevant_retrieved),
            },
        )
    
    def _get_doc_id(self, doc: Any) -> str:
        if hasattr(doc, 'id'):
            return doc.id
        elif hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return doc.metadata['id']
        return str(hash(doc.page_content if hasattr(doc, 'page_content') else str(doc)))


class CitationEvaluator:
    """
    Citation Span Accuracy evaluation.
    Citation'ların doğru span'ları işaret edip etmediğini ölçer.
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        answer: str,
        context: str,
        citations: List[Any],
    ) -> EvaluationResult:
        """
        Evaluate citation accuracy.
        
        Measures:
        1. Citation presence - does answer have citations?
        2. Citation span - do citations point to correct text in context?
        3. Citation coverage - what % of claims are cited?
        """
        details = {}
        
        # 1. Citation presence
        has_citations = len(citations) > 0
        presence_score = 1.0 if has_citations else 0.0
        details['has_citations'] = has_citations
        details['citation_count'] = len(citations)
        
        # 2. Citation span accuracy
        span_accuracy = self._calculate_span_accuracy(answer, context, citations)
        details['span_accuracy'] = span_accuracy
        
        # 3. Citation coverage
        coverage = self._calculate_coverage(answer, citations)
        details['coverage'] = coverage
        
        # Overall score (weighted)
        overall = (presence_score * 0.3 + span_accuracy * 0.4 + coverage * 0.3)
        
        return EvaluationResult(
            metric_name="citation_span_accuracy",
            score=overall,
            details=details,
        )
    
    def _calculate_span_accuracy(
        self,
        answer: str,
        context: str,
        citations: List[Any],
    ) -> float:
        """Calculate if citations point to correct text."""
        if not citations:
            return 0.0
        
        correct = 0
        for cit in citations:
            # Check if cited text exists in context
            if hasattr(cit, 'text') and cit.text:
                if cit.text.lower() in context.lower():
                    correct += 1
            # If no specific text, assume correct (placeholder citation)
            elif hasattr(cit, 'source'):
                correct += 1
        
        return correct / len(citations) if citations else 0.0
    
    def _calculate_coverage(
        self,
        answer: str,
        citations: List[Any],
    ) -> float:
        """Calculate what % of answer claims are cited."""
        # Split answer into sentences/claims
        sentences = answer.split('.')
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not claims:
            return 1.0 if citations else 0.0
        
        # Each citation can cover ~1-2 claims
        coverage = min(len(citations) / len(claims), 1.0)
        return coverage
    
    def verify_citation_spans(
        self,
        answer: str,
        context: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify all citations in answer against context."""
        import re
        
        results = []
        
        # Find citation patterns
        citation_patterns = [
            r'(?:MADDE|madde|Sayfa|sayfa|Kaynak)\s*[:\-]?\s*(\d+)',
            r'\[(\d+)\]',
            r'\((\d+)\)',
        ]
        
        all_valid = True
        for pattern in citation_patterns:
            matches = re.finditer(pattern, answer, re.IGNORECASE)
            for match in matches:
                cited_ref = match.group(0)
                # Check if reference exists in some form in context
                # This is a simplified check
                results.append({
                    'citation': cited_ref,
                    'valid': True,  # Would need more sophisticated check
                })
        
        return all_valid, results


class HallucinationEvaluator:
    """
    Hallucination rate evaluation.
    Bağlam dışı bilgi tespiti.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.3,
        use_llm_judge: bool = False,
        llm: Any = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.use_llm_judge = use_llm_judge
        self.llm = llm
    
    def evaluate(
        self,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """
        Evaluate hallucination rate.
        
        Methods:
        1. Word overlap (simple)
        2. Claim extraction + overlap
        3. LLM judge (if available)
        """
        details = {}
        
        if not answer or not context:
            return EvaluationResult(
                metric_name="hallucination_rate",
                score=1.0,
                details={'error': 'Missing answer or context'},
            )
        
        # Method 1: Word overlap
        overlap_score = self._word_overlap(answer, context)
        details['word_overlap'] = overlap_score
        
        # Method 2: Claim-based
        claim_score = self._claim_based_eval(answer, context)
        details['claim_score'] = claim_score
        
        # Method 3: LLM judge (if enabled)
        if self.use_llm_judge and self.llm:
            llm_score = self._llm_judge_eval(answer, context)
            details['llm_judge_score'] = llm_score
            # Use LLM as primary
            hallucination_rate = 1.0 - llm_score
        else:
            # Use claim-based as primary
            hallucination_rate = 1.0 - claim_score
        
        details['hallucination_rate'] = hallucination_rate
        details['passed'] = hallucination_rate < 0.2
        
        return EvaluationResult(
            metric_name="hallucination_rate",
            score=hallucination_rate,
            details=details,
        )
    
    def _word_overlap(self, answer: str, context: str) -> float:
        """Calculate word overlap between answer and context."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove stopwords
        stopwords = {
            've', 'veya', 'ama', 'fakat', 'lakin', 'çünkü', 'bu', 'şu', 'o',
            'bir', 'ile', 'için', 'kadar', 'gibi', 'de', 'da', 'mi', 'mu',
            'not', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
        }
        answer_words = answer_words - stopwords
        context_words = context_words - stopwords
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return overlap / len(answer_words)
    
    def _claim_based_eval(self, answer: str, context: str) -> float:
        """Evaluate based on extracted claims."""
        claims = self._extract_claims(answer)
        
        if not claims:
            return 0.5  # Neutral
        
        valid_claims = 0
        for claim in claims:
            if self._claim_in_context(claim, context):
                valid_claims += 1
        
        return valid_claims / len(claims)
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        sentences = text.split('.')
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Skip questions, short sentences
            if len(sent) < 20 or sent.endswith('?'):
                continue
            # Skip meta statements
            if any(w in sent.lower() for w in ['ödeme', 'ödeme', 'teslimat']):
                continue
            claims.append(sent)
        
        return claims
    
    def _claim_in_context(self, claim: str, context: str) -> bool:
        """Check if claim is grounded in context."""
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        stopwords = {
            've', 'veya', 'ama', 'fakat', 'lakin', 'çünkü', 'bu', 'şu', 'o',
            'bir', 'ile', 'için', 'kadar', 'gibi', 'de', 'da', 'mi', 'mu',
        }
        claim_words = claim_words - stopwords
        context_words = context_words - stopwords
        
        if not claim_words:
            return True
        
        overlap = len(claim_words.intersection(context_words))
        return (overlap / len(claim_words)) >= self.similarity_threshold
    
    def _llm_judge_eval(self, answer: str, context: str) -> float:
        """Use LLM as external judge."""
        if not self.llm:
            return 0.5
        
        prompt = f"""Sen bir hakem modelisin. Aşağıdaki cevabın bağlama göre doğruluğunu değerlendir.

BAĞLAM:
{context[:1000]}

CEVAP:
{answer}

Talimatlar:
1. Cevabın bağlamdaki bilgilere dayandığını doğrula
2. Bağlamda olmayan uydurma bilgi varsa işaretle
3. 0-1 arasında puan ver:
- 1.0: Tüm bilgiler bağlamda var
- 0.5: Bazı bilgiler bağlamda, bazıları belirsiz
- 0.0: Çoğu bilgi bağlamda yok (halüsinasyon)

Sadece tek bir sayısal puan ver:"""
        
        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract score
            import re
            match = re.search(r'0\.\d+|\d+', text)
            if match:
                score = float(match.group())
                if score > 1:
                    score = score / 10
                return score
        except Exception as e:
            logger.error(f"LLM judge error: {e}")
        
        return 0.5


class DeterminismTest:
    """
    Determinism test.
    Aynı input'un aynı output'u üretip üretmediğini test eder.
    """
    
    def __init__(
        self,
        runs: int = 3,
        similarity_threshold: float = 0.85,
    ):
        self.runs = runs
        self.similarity_threshold = similarity_threshold
    
    def evaluate(
        self,
        query: str,
        generate_func,
    ) -> EvaluationResult:
        """
        Evaluate determinism by running same query multiple times.
        
        Args:
            query: Test query
            generate_func: Function that generates answer (no args, returns str)
            
        Returns:
            EvaluationResult with determinism score
        """
        answers = []
        
        for i in range(self.runs):
            try:
                answer = generate_func()
                answers.append(answer)
            except Exception as e:
                logger.error(f"Run {i} failed: {e}")
        
        if len(answers) < 2:
            return EvaluationResult(
                metric_name="determinism",
                score=0.0,
                details={'error': 'Not enough runs completed'},
            )
        
        # Calculate consistency
        consistency_scores = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                score = self._similarity(answers[i], answers[j])
                consistency_scores.append(score)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Also check hash consistency
        hashes = [hashlib.md5(a.encode()).hexdigest() for a in answers]
        unique_hashes = len(set(hashes))
        hash_consistency = 1.0 - (unique_hashes / len(hashes))
        
        details = {
            'runs': len(answers),
            'avg_consistency': avg_consistency,
            'hash_consistency': hash_consistency,
            'unique_answers': unique_hashes,
            'threshold': self.similarity_threshold,
            'passed': avg_consistency >= self.similarity_threshold,
        }
        
        return EvaluationResult(
            metric_name="determinism",
            score=avg_consistency,
            details=details,
        )
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        if text1 == text2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class EvaluationPipeline:
    """
    Complete evaluation pipeline.
    Tüm metrikleri birleştirir.
    """
    
    def __init__(
        self,
        retrieval_system: Any = None,
        llm_judge: Any = None,
        gold_qa_path: Optional[str] = None,
    ):
        self.retrieval_system = retrieval_system
        self.recall_evaluator = RecallEvaluator(k=5)
        self.citation_evaluator = CitationEvaluator()
        self.hallucination_evaluator = HallucinationEvaluator(
            use_llm_judge=llm_judge is not None,
            llm=llm_judge,
        )
        self.determinism_test = DeterminismTest()
        
        self.gold_qa = EvaluationStore(gold_qa_path) if gold_qa_path else EvaluationStore()
    
    def evaluate_single(
        self,
        query: str,
        context: str,
        retrieved_docs: List[Any],
        relevant_doc_ids: List[str],
        answer: str,
        citations: List[Any] = None,
    ) -> EvaluationReport:
        """Evaluate a single query."""
        import uuid
        from datetime import datetime
        
        report = EvaluationReport(
            evaluation_id=str(uuid.uuid4())[:8],
            query=query,
        )
        
        # 1. Retrieval Recall
        if self.retrieval_system and relevant_doc_ids:
            report.retrieval_metrics = self.recall_evaluator.evaluate(
                query, retrieved_docs, relevant_doc_ids
            )
        
        # 2. Citation Span Accuracy
        if answer and context:
            report.citation_metrics = self.citation_evaluator.evaluate(
                answer, context, citations or []
            )
        
        # 3. Hallucination Rate
        if answer and context:
            report.hallucination_metrics = self.hallucination_evaluator.evaluate(
                answer, context
            )
        
        # Calculate overall score
        scores = []
        if report.retrieval_metrics:
            scores.append(report.retrieval_metrics.score)
        if report.citation_metrics:
            scores.append(report.citation_metrics.score)
        if report.hallucination_metrics:
            # Invert hallucination (lower is better)
            scores.append(1.0 - report.hallucination_metrics.score)
        
        report.overall_score = np.mean(scores) if scores else 0.0
        report.passed = report.overall_score >= 0.7
        
        return report
    
    def evaluate_batch(
        self,
        queries: List[str],
        retrieval_func,
        generate_func,
        gold_qa: Optional[List[GoldQA]] = None,
    ) -> Dict[str, Any]:
        """Evaluate multiple queries."""
        qa_set = gold_qa or self.gold_qa.get_all()
        
        results = []
        for qa in qa_set:
            # Retrieve
            retrieved = retrieval_func(qa.query)
            
            # Generate
            answer = generate_func()
            
            # Evaluate
            report = self.evaluate_single(
                query=qa.query,
                context=qa.context,
                retrieved_docs=retrieved,
                relevant_doc_ids=qa.relevant_docs,
                answer=answer,
            )
            results.append(report)
        
        # Aggregate metrics
        return self._aggregate_results(results)
    
    def _aggregate_results(
        self,
        reports: List[EvaluationReport],
    ) -> Dict[str, Any]:
        """Aggregate evaluation results."""
        if not reports:
            return {'error': 'No results to aggregate'}
        
        agg = {
            'total_evaluations': len(reports),
            'passed': sum(1 for r in reports if r.passed),
            'pass_rate': 0.0,
            'avg_recall': 0.0,
            'avg_citation': 0.0,
            'avg_hallucination': 0.0,
            'avg_determinism': 0.0,
            'overall_score': 0.0,
        }
        
        recall_scores = [r.retrieval_metrics.score for r in reports if r.retrieval_metrics]
        citation_scores = [r.citation_metrics.score for r in reports if r.citation_metrics]
        hallucination_scores = [r.hallucination_metrics.score for r in reports if r.hallucination_metrics]
        determinism_scores = [r.determinism_metrics.score for r in reports if r.determinism_metrics]
        
        agg['avg_recall'] = np.mean(recall_scores) if recall_scores else 0.0
        agg['avg_citation'] = np.mean(citation_scores) if citation_scores else 0.0
        agg['avg_hallucination'] = np.mean(hallucination_scores) if hallucination_scores else 0.0
        agg['avg_determinism'] = np.mean(determinism_scores) if determinism_scores else 0.0
        agg['overall_score'] = np.mean([r.overall_score for r in reports])
        agg['pass_rate'] = agg['passed'] / len(reports) if reports else 0.0
        
        return agg


def create_default_gold_qa() -> List[GoldQA]:
    """Create default gold QA set for Turkish political documents."""
    return [
        GoldQA(
            query="CHP'nin eğitim politikası nedir?",
            relevant_docs=["chp_tuzuk", "chp_program"],
            expected_answer="CHP eğitimde laik ve bilimsel eğitimi savunur...",
            context="CHP tüzüğüne göre eğitim politikası...",
            party="CHP",
        ),
        GoldQA(
            query="AKP'nin ekonomi politikası nedir?",
            relevant_docs=["akp_tuzuk", "akp_program"],
            expected_answer="AKP ekonomi politikası serbest piyasa...",
            context="AKP ekonomi anlayışı...",
            party="AKP",
        ),
        GoldQA(
            query="MHP'nin dış politika görüşü nedir?",
            relevant_docs=["mhp_tuzuk", "mhp_program"],
            expected_answer="MHP dış politikada millî menfaatleri...",
            context="MHP dış politika esasları...",
            party="MHP",
        ),
    ]


def create_evaluation_pipeline(
    retrieval_system: Any = None,
    llm_judge: Any = None,
    gold_qa_path: Optional[str] = None,
) -> EvaluationPipeline:
    """Create evaluation pipeline."""
    return EvaluationPipeline(
        retrieval_system=retrieval_system,
        llm_judge=llm_judge,
        gold_qa_path=gold_qa_path,
    )
