from functools import lru_cache
from typing import Any, Optional, Tuple
import logging
import config
import utils

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_embeddings() -> Any:
    """
    Load and cache the embedding model (singleton pattern).
    Uses lru_cache for efficiency.
    """
    try:
        embeddings = utils.load_embeddings()
        logger.info("✅ Embeddings loaded and cached")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Embedding load error: {str(e)}")
        raise


def get_vectorstore() -> Any:
    """
    Lazy load the unified vectorstore (singleton pattern).
    Uses cached embeddings.
    """
    embeddings = get_cached_embeddings()
    return utils.load_vectorstore(config.UNIFIED_VECTOR_DB, embeddings)


def get_party_vectorstore(party: str) -> Any:
    """
    Load vectorstore for a specific party.
    Uses the unified DB with metadata filtering.
    """
    embeddings = get_cached_embeddings()
    return utils.load_vectorstore(config.UNIFIED_VECTOR_DB, embeddings)
