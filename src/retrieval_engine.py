"""
FAZ 2: Retrieval Engine
Hybrid + optimize edilmiş retrieval sistemi.

Bu modül:
- Dense retrieval (embedding tabanlı)
- Sparse retrieval (BM25)
- Hybrid score fusion
- Retrieval evaluation (Recall@k)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Retrieval sonucu."""
    query: str
    documents: List[Any] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    method: str = "unknown"
    latency_ms: float = 0.0
    
    @property
    def top_doc(self) -> Optional[Any]:
        """En yüksek skorlu doküman."""
        return self.documents[0] if self.documents else None
    
    @property
    def top_score(self) -> float:
        """En yüksek skor."""
        return self.scores[0] if self.scores else 0.0


@dataclass
class RetrievalMetrics:
    """Retrieval metrikleri."""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'recall_at_k': self.recall_at_k,
            'precision_at_k': self.precision_at_k,
            'mrr': self.mrr,
            'ndcg': self.ndcg,
            'latency_ms': self.latency_ms,
        }


class BaseRetrieval(ABC):
    """Abstract base class for retrieval methods."""
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Search documents."""
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Any]:
        """Tüm dokümanları getir."""
        pass


class DenseRetrieval(BaseRetrieval):
    """
    Dense retrieval - Embedding tabanlı.
    Chroma vector store kullanır.
    """
    
    def __init__(self, vectorstore: Any, embeddings: Any):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Dense search (cosine similarity)."""
        import time
        start = time.time()
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filter_metadata
            )
            
            docs = [doc for doc, score in results]
            scores = [score for doc, score in results]
            
            # Chroma scores are distances (lower = better), convert to similarity
            # For Chroma with cosine: higher = more similar
            # But we get negative distances, so let's normalize
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                if max_score != min_score:
                    scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    scores = [1.0] * len(scores)
            
            latency = (time.time() - start) * 1000
            
            return RetrievalResult(
                query=query,
                documents=docs,
                scores=scores,
                method="dense",
                latency_ms=latency,
            )
            
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return RetrievalResult(query=query, method="dense")
    
    def get_all_documents(self) -> List[Any]:
        """Tüm dokümanları getir."""
        return self.vectorstore.get()['documents']


class SparseRetrieval(BaseRetrieval):
    """
    Sparse retrieval - BM25 algoritması.
    Rank-BM25 kütüphanesini kullanır.
    """
    
    def __init__(self, documents: List[str], document_ids: Optional[List[str]] = None):
        """
        Args:
            documents: Doküman metinleri
            document_ids: Doküman ID'leri (opsiyonel)
        """
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(documents)
            self.document_ids = document_ids or [f"doc_{i}" for i in range(len(documents))]
            self.documents = documents
            self._doc_lookup = {i: doc for i, doc in enumerate(documents)}
        except ImportError:
            logger.warning("rank_bm25 not installed, falling back to dummy")
            self.bm25 = None
            self.documents = documents
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """BM25 search."""
        import time
        start = time.time()
        
        if self.bm25 is None:
            logger.warning("BM25 not available, returning empty results")
            return RetrievalResult(query=query, method="sparse")
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            docs = [self._doc_lookup[i] for i in top_indices]
            bm25_scores = [scores[i] for i in top_indices]
            
            # Normalize scores to [0, 1]
            if bm25_scores:
                max_score = max(bm25_scores)
                if max_score > 0:
                    bm25_scores = [s / max_score for s in bm25_scores]
            
            latency = (time.time() - start) * 1000
            
            return RetrievalResult(
                query=query,
                documents=docs,
                scores=bm25_scores,
                method="sparse",
                latency_ms=latency,
            )
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return RetrievalResult(query=query, method="sparse")
    
    def get_all_documents(self) -> List[str]:
        """Tüm dokümanları getir."""
        return self.documents
    
    @classmethod
    def from_vectorstore(cls, vectorstore: Any) -> 'SparseRetrieval':
        """Vector store'dan BM25 oluştur."""
        docs = vectorstore.get()
        documents = docs.get('documents', [])
        doc_ids = docs.get('ids', None)
        return cls(documents, doc_ids)


class HybridRetrieval(BaseRetrieval):
    """
    Hybrid retrieval - Dense + Sparse fusion.
    Score fusion yöntemleri: RRF, weighted sum, combMNZ
    """
    
    def __init__(
        self, 
        dense_retrieval: DenseRetrieval,
        sparse_retrieval: SparseRetrieval,
        fusion_method: str = "rrf",  # rrf, weighted, combMNZ
        dense_weight: float = 0.5,
        k: int = 60,  # RRF parameter
    ):
        self.dense = dense_retrieval
        self.sparse = sparse_retrieval
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.k = k
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Hybrid search with score fusion."""
        import time
        start = time.time()
        
        # Parallel search
        dense_result = self.dense.search(query, top_k * 2, filter_metadata)
        sparse_result = self.sparse.search(query, top_k * 2, filter_metadata)
        
        # Fusion
        fused_scores, fused_docs = self._fuse_results(
            dense_result, sparse_result, top_k
        )
        
        latency = (time.time() - start) * 1000
        
        return RetrievalResult(
            query=query,
            documents=fused_docs,
            scores=fused_scores,
            method=f"hybrid_{self.fusion_method}",
            latency_ms=latency,
        )
    
    def _fuse_results(
        self, 
        dense: RetrievalResult, 
        sparse: RetrievalResult,
        top_k: int
    ) -> Tuple[List[float], List[Any]]:
        """Fuse scores from both retrievers."""
        
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense, sparse, top_k)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(dense, sparse, top_k)
        elif self.fusion_method == "combMNZ":
            return self._combMNZ_fusion(dense, sparse, top_k)
        else:
            return self._reciprocal_rank_fusion(dense, sparse, top_k)
    
    def _reciprocal_rank_fusion(
        self, 
        dense: RetrievalResult, 
        sparse: RetrievalResult,
        top_k: int
    ) -> Tuple[List[float], List[Any]]:
        """Reciprocal Rank Fusion."""
        
        # Build score map
        doc_scores = {}
        
        # Dense scores
        for i, doc in enumerate(dense.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
            doc_scores[doc_id]['dense'] = dense.scores[i] if i < len(dense.scores) else 0
            doc_scores[doc_id]['doc'] = doc
        
        # Sparse scores
        for i, doc in enumerate(sparse.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {}
            doc_scores[doc_id]['sparse'] = sparse.scores[i] if i < len(sparse.scores) else 0
            doc_scores[doc_id]['doc'] = doc
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id, scores_dict in doc_scores.items():
            rrf = 0.0
            
            # Dense RRF
            if 'dense' in scores_dict and scores_dict['dense'] > 0:
                rank_dense = list(dense.documents).index(scores_dict['doc']) + 1
                rrf += 1.0 / (self.k + rank_dense) * self.dense_weight
            
            # Sparse RRF
            if 'sparse' in scores_dict and scores_dict['sparse'] > 0:
                rank_sparse = list(sparse.documents).index(scores_dict['doc']) + 1
                rrf += 1.0 / (self.k + rank_sparse) * self.sparse_weight
            
            rrf_scores[doc_id] = rrf
        
        # Sort and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_scores = [score for _, score in sorted_docs]
        final_docs = [doc for doc_id, score in sorted_docs for doc_id_key, doc in doc_scores.items() if doc_id_key == doc_id]
        
        return final_scores, final_docs
    
    def _weighted_fusion(
        self, 
        dense: RetrievalResult, 
        sparse: RetrievalResult,
        top_k: int
    ) -> Tuple[List[float], List[Any]]:
        """Weighted score fusion."""
        
        doc_scores = {}
        
        for i, doc in enumerate(dense.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'dense': 0, 'sparse': 0}
            doc_scores[doc_id]['dense'] = dense.scores[i] if i < len(dense.scores) else 0
        
        for i, doc in enumerate(sparse.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'dense': 0, 'sparse': 0}
            doc_scores[doc_id]['sparse'] = sparse.scores[i] if i < len(sparse.scores) else 0
        
        # Weighted sum
        for doc_id in doc_scores:
            doc_scores[doc_id]['combined'] = (
                doc_scores[doc_id]['dense'] * self.dense_weight +
                doc_scores[doc_id]['sparse'] * self.sparse_weight
            )
        
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1]['combined'], 
            reverse=True
        )[:top_k]
        
        return [d[1]['combined'] for d in sorted_docs], [d[1]['doc'] for d in sorted_docs]
    
    def _combMNZ_fusion(
        self, 
        dense: RetrievalResult, 
        sparse: RetrievalResult,
        top_k: int
    ) -> Tuple[List[float], List[Any]]:
        """Combination with Multiple Normalized scores."""
        
        doc_scores = {}
        
        for i, doc in enumerate(dense.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'scores': [], 'count': 0}
            doc_scores[doc_id]['scores'].append(dense.scores[i] if i < len(dense.scores) else 0)
            doc_scores[doc_id]['count'] += 1
        
        for i, doc in enumerate(sparse.documents):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'scores': [], 'count': 0}
            doc_scores[doc_id]['scores'].append(sparse.scores[i] if i < len(sparse.scores) else 0)
            doc_scores[doc_id]['count'] += 1
        
        # CombMNZ: sum of normalized scores * number of retrievers
        for doc_id in doc_scores:
            avg_score = sum(doc_scores[doc_id]['scores']) / len(doc_scores[doc_id]['scores'])
            doc_scores[doc_id]['combined'] = avg_score * doc_scores[doc_id]['count']
        
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1]['combined'], 
            reverse=True
        )[:top_k]
        
        return [d[1]['combined'] for d in sorted_docs], [d[1]['doc'] for d in sorted_docs]
    
    def _get_doc_id(self, doc: Any) -> str:
        """Doküman ID'sini al."""
        if hasattr(doc, 'id'):
            return doc.id
        elif hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return doc.metadata['id']
        else:
            return str(hash(doc.page_content if hasattr(doc, 'page_content') else str(doc)))
    
    def get_all_documents(self) -> List[Any]:
        """Tüm dokümanları getir."""
        return self.dense.get_all_documents()


class RetrievalEvaluator:
    """
    Retrieval evaluation agent.
    Recall@k, precision, MRR, NDCG ölçer.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(
        self, 
        retrieved_docs: List[Any], 
        relevant_docs: List[str]
    ) -> RetrievalMetrics:
        """
        Retrieval performansını değerlendirir.
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_docs: Ground truth relevant document IDs/contents
            
        Returns:
            RetrievalMetrics: Evaluation metrics
        """
        metrics = RetrievalMetrics()
        
        if not retrieved_docs or not relevant_docs:
            return metrics
        
        # Convert to sets for comparison
        retrieved_ids = set()
        for doc in retrieved_docs[:self.k]:
            doc_id = self._get_doc_id(doc)
            retrieved_ids.add(doc_id)
        
        relevant_set = set(relevant_docs)
        
        # Recall@K
        relevant_retrieved = retrieved_ids.intersection(relevant_set)
        metrics.recall_at_k = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        
        # Precision@K
        metrics.precision_at_k = len(relevant_retrieved) / min(len(retrieved_ids), self.k) if retrieved_ids else 0.0
        
        # MRR (Mean Reciprocal Rank)
        for i, doc in enumerate(retrieved_docs[:self.k]):
            doc_id = self._get_doc_id(doc)
            if doc_id in relevant_set:
                metrics.mrr = 1.0 / (i + 1)
                break
        
        # NDCG (Normalized Discounted Cumulative Gain)
        metrics.ndcg = self._calculate_ndcg(retrieved_docs, relevant_set)
        
        return metrics
    
    def evaluate_batch(
        self, 
        queries: List[str], 
        retrieval_func, 
        ground_truth: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Batch evaluation.
        
        Args:
            queries: Test queries
            retrieval_func: Retrieval function(query) -> List[docs]
            ground_truth: {query: [relevant_doc_ids]}
            
        Returns:
            Dict: Average metrics
        """
        all_metrics = []
        
        for query in queries:
            retrieved = retrieval_func(query)
            relevant = ground_truth.get(query, [])
            metrics = self.evaluate(retrieved, relevant)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            'avg_recall': np.mean([m.recall_at_k for m in all_metrics]),
            'avg_precision': np.mean([m.precision_at_k for m in all_metrics]),
            'avg_mrr': np.mean([m.mrr for m in all_metrics]),
            'avg_ndcg': np.mean([m.ndcg for m in all_metrics]),
        }
        
        return avg_metrics
    
    def _calculate_ndcg(self, retrieved_docs: List[Any], relevant_set: set) -> float:
        """Calculate NDCG."""
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:self.k]):
            doc_id = self._get_doc_id(doc)
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Ideal DCG
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), self.k))])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get document ID."""
        if hasattr(doc, 'id'):
            return doc.id
        elif hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return doc.metadata['id']
        else:
            return str(hash(doc.page_content if hasattr(doc, 'page_content') else str(doc)))


def create_hybrid_retrieval(
    vectorstore: Any,
    embeddings: Any,
    fusion_method: str = "rrf",
    dense_weight: float = 0.5,
) -> HybridRetrieval:
    """
    Hybrid retrieval factory.
    
    Args:
        vectorstore: Chroma vector store
        embeddings: Embedding model
        fusion_method: rrf, weighted, combMNZ
        dense_weight: Weight for dense retrieval (sparse = 1 - dense_weight)
        
    Returns:
        HybridRetrieval: Hybrid retrieval instance
    """
    dense = DenseRetrieval(vectorstore, embeddings)
    sparse = SparseRetrieval.from_vectorstore(vectorstore)
    
    return HybridRetrieval(
        dense_retrieval=dense,
        sparse_retrieval=sparse,
        fusion_method=fusion_method,
        dense_weight=dense_weight,
    )
