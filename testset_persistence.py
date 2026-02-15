"""
Test Dataset Persistence for Ragas
Caches generated test datasets to avoid expensive LLM calls
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Any, Dict, Optional
from datetime import datetime
from ragas.testset import TestsetGenerator
from langchain_core.documents import Document
from loguru import logger


# ============================================================================
# CORE PERSISTENCE CLASS
# ============================================================================

class TestDatasetCache:
    """
    Smart caching for Ragas test datasets.
    Avoids expensive LLM calls by persisting generated datasets.
    """
    
    def __init__(self, cache_dir: str = "./testset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "testset_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_cache_key(
        self, 
        docs: List[Any], 
        testset_size: int,
        generator_llm_name: Optional[str] = None
    ) -> str:
        """
        Generate unique cache key based on:
        - Document contents (hash)
        - Number of documents
        - Test set size
        - Generator LLM model (optional)
        """
        # Create hash of document contents
        doc_contents = [doc.page_content for doc in docs]
        doc_hash = hashlib.md5(
            "".join(doc_contents).encode()
        ).hexdigest()
        
        # Combine parameters
        key_string = f"{doc_hash}_{len(docs)}_{testset_size}"
        if generator_llm_name:
            key_string += f"_{generator_llm_name}"
        
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        return cache_key
    
    def _save_dataset(
        self, 
        cache_key: str, 
        dataset: Any,
        docs: List[Any],
        testset_size: int,
        generator_llm_name: Optional[str] = None
    ):
        """Save dataset to cache"""
        # Save the actual dataset
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Also save as JSON for human readability
        json_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Convert dataset to serializable format
            dataset_dict = self._dataset_to_dict(dataset)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save JSON version: {e}")
        
        # Update metadata
        self.metadata[cache_key] = {
            'created_at': datetime.now().isoformat(),
            'num_docs': len(docs),
            'testset_size': testset_size,
            'generator_llm': generator_llm_name,
            'cache_file': str(cache_file),
            'json_file': str(json_file)
        }
        self._save_metadata()
    
    def _load_dataset(self, cache_key: str) -> Any:
        """Load dataset from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def _dataset_to_dict(self, dataset: Any) -> Dict:
        """Convert Ragas dataset to dictionary for JSON serialization"""
        if hasattr(dataset, 'to_pandas'):
            # If it's a Ragas Dataset, convert to pandas then dict
            df = dataset.to_pandas()
            return {
                'type': 'ragas_dataset',
                'data': df.to_dict(orient='records'),
                'columns': list(df.columns)
            }
        elif hasattr(dataset, '__dict__'):
            # Try to extract data from object
            return {
                'type': 'object',
                'data': str(dataset)
            }
        else:
            return {'type': 'unknown', 'data': str(dataset)}
    
    def generate_with_cache(
        self,
        generator_llm: Any,
        generator_embeddings: Any,
        docs: List[Any],
        testset_size: int = 10,
        force_regenerate: bool = False,
        llm_name: Optional[str] = None
    ) -> Any:
        """
        Generate test dataset with caching.
        
        Args:
            generator_llm: LangchainLLMWrapper for generation
            generator_embeddings: LangchainEmbeddingsWrapper for embeddings
            docs: List of documents to generate questions from
            testset_size: Number of test samples to generate
            force_regenerate: If True, ignore cache and regenerate
            llm_name: Optional name of LLM model for cache key
        
        Returns:
            Generated Ragas dataset
        """
        # Generate cache key
        cache_key = self._generate_cache_key(docs, testset_size, llm_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if not force_regenerate and cache_file.exists():
            logger.info(f"Loading test dataset from cache (key: {cache_key[:8]}...)")
            try:
                dataset = self._load_dataset(cache_key)
                metadata = self.metadata.get(cache_key, {})
                logger.info(f"Created: {metadata.get('created_at', 'Unknown')}")
                logger.info(f"Size: {testset_size} samples")
                logger.info(f"Docs: {len(docs)}")
                return dataset
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Regenerating...")
        
        # Generate new dataset
        logger.info("Generating new test dataset...")
        logger.info(f"Documents: {len(docs)}")
        logger.info(f"Test size: {testset_size}")
        logger.info("This may take a few minutes...")
        
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        
        dataset = generator.generate_with_langchain_docs(
            docs, 
            testset_size=testset_size
        )
        
        # Save to cache
        logger.success("Test dataset generated successfully")
        self._save_dataset(cache_key, dataset, docs, testset_size, llm_name)
        logger.success(f"Saved to cache (key: {cache_key[:8]}...)")
        
        return dataset
    
    def list_cached_datasets(self):
        """List all cached test datasets"""
        if not self.metadata:
            logger.info("No cached datasets found.")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("CACHED TEST DATASETS")
        logger.info("=" * 80)
        
        for cache_key, info in self.metadata.items():
            cache_file = Path(info['cache_file'])
            if cache_file.exists():
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                logger.info(f"\nCache Key: {cache_key[:12]}...")
                logger.info(f"  Created: {info['created_at']}")
                logger.info(f"  Documents: {info['num_docs']}")
                logger.info(f"  Test Size: {info['testset_size']}")
                logger.info(f"  LLM: {info.get('generator_llm', 'N/A')}")
                logger.info(f"  File Size: {size_mb:.2f} MB")
                logger.info(f"  Files:")
                logger.info(f"    - {info['cache_file']}")
                if 'json_file' in info:
                    logger.info(f"    - {info['json_file']}")
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache (specific key or all)"""
        if cache_key:
            # Clear specific cache
            if cache_key in self.metadata:
                info = self.metadata[cache_key]
                for file_key in ['cache_file', 'json_file']:
                    if file_key in info:
                        file_path = Path(info[file_key])
                        if file_path.exists():
                            file_path.unlink()
                del self.metadata[cache_key]
                self._save_metadata()
                logger.success(f"Cleared cache: {cache_key[:12]}...")
            else:
                logger.warning(f"Cache key not found: {cache_key}")
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for json_file in self.cache_dir.glob("*.json"):
                if json_file != self.metadata_file:
                    json_file.unlink()
            self.metadata = {}
            self._save_metadata()
            logger.success("All caches cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached datasets"""
        total_size = 0
        valid_caches = 0
        
        for cache_key, info in self.metadata.items():
            cache_file = Path(info['cache_file'])
            if cache_file.exists():
                total_size += cache_file.stat().st_size
                valid_caches += 1
        
        return {
            'total_cached': len(self.metadata),
            'valid_caches': valid_caches,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


# ============================================================================
# CONVENIENCE FUNCTION (Drop-in Replacement)
# ============================================================================

def generate_test_dataset_cached(
    generator_llm: Any,
    generator_embeddings: Any,
    docs: List[Any],
    testset_size: int = 10,
    cache_dir: str = "./testset_cache",
    force_regenerate: bool = False,
    llm_name: Optional[str] = None
) -> Any:
    """
    Drop-in replacement for generate_test_dataset with caching.
    
    Args:
        generator_llm: LangchainLLMWrapper for generation
        generator_embeddings: LangchainEmbeddingsWrapper for embeddings
        docs: List of documents to generate questions from
        testset_size: Number of test samples to generate
        cache_dir: Directory to store cached datasets
        force_regenerate: If True, ignore cache and regenerate
        llm_name: Optional name of LLM model for cache key
    
    Returns:
        Generated Ragas dataset
    """
    cache = TestDatasetCache(cache_dir=cache_dir)
    return cache.generate_with_cache(
        generator_llm=generator_llm,
        generator_embeddings=generator_embeddings,
        docs=docs,
        testset_size=testset_size,
        force_regenerate=force_regenerate,
        llm_name=llm_name
    )


# ============================================================================
# ORIGINAL FUNCTION (For Reference)
# ============================================================================

def generate_test_dataset(
    generator_llm: Any, 
    generator_embeddings: Any,
    docs: List[Any],
    testset_size: int = 10
) -> Dict[str, Any]:
    """Original function - NOT cached"""
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    return dataset


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Basic usage example"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Setup (same as before)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Your documents
    docs = [
        Document(page_content="AI is transforming industries...", metadata={"id": 1}),
        Document(page_content="Machine learning requires data...", metadata={"id": 2}),
        # ... more docs
    ]
    
    # METHOD 1: Simple function call (recommended)
    logger.info("\n" + "=" * 80)
    logger.info("METHOD 1: Simple Function Call")
    logger.info("=" * 80)
    
    dataset = generate_test_dataset_cached(
        generator_llm=generator_llm,
        generator_embeddings=generator_embeddings,
        docs=docs,
        testset_size=10,
        cache_dir="./testset_cache"
    )
    
    logger.success(f"Got dataset with {len(dataset)} samples")


def example_cache_management():
    """Example of cache management"""
    
    # Initialize cache
    cache = TestDatasetCache(cache_dir="./testset_cache")
    
    # List cached datasets
    cache.list_cached_datasets()
    
    # Get cache statistics
    stats = cache.get_cache_stats()
    logger.info("\nCache Statistics:")
    logger.info(f"  Total cached: {stats['total_cached']}")
    logger.info(f"  Valid caches: {stats['valid_caches']}")
    logger.info(f"  Total size: {stats['total_size_mb']:.2f} MB")
    logger.info(f"  Cache directory: {stats['cache_dir']}")
    
    # Clear specific cache (optional)
    # cache.clear_cache(cache_key="abc123...")
    
    # Clear all caches (optional)
    # cache.clear_cache()


def example_with_different_llms():
    """Example showing how to cache for different LLMs"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    cache = TestDatasetCache()
    
    docs = [...]  # Your documents
    
    # Generate with GPT-3.5
    llm_gpt35 = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    dataset_gpt35 = cache.generate_with_cache(
        generator_llm=llm_gpt35,
        generator_embeddings=embeddings,
        docs=docs,
        testset_size=10,
        llm_name="gpt-3.5-turbo"  # Include model name in cache key
    )
    
    # Generate with GPT-4 (will create separate cache)
    llm_gpt4 = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    
    dataset_gpt4 = cache.generate_with_cache(
        generator_llm=llm_gpt4,
        generator_embeddings=embeddings,
        docs=docs,
        testset_size=10,
        llm_name="gpt-4"  # Different cache key
    )


# ============================================================================
# INTEGRATION WITH YOUR RETRIEVER EVALUATION
# ============================================================================

def integrated_example():
    """
    Complete example integrating with retriever evaluation
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    logger.info("=" * 80)
    logger.info("INTEGRATED RETRIEVER EVALUATION WITH CACHED TEST DATASET")
    logger.info("=" * 80)
    
    # Step 1: Load documents (use your DatasetCache from previous artifact)
    # docs = load_documents(...)
    
    # Step 2: Generate test dataset (CACHED!)
    cache = TestDatasetCache(cache_dir="./testset_cache")
    
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # This will be instant on subsequent runs!
    test_dataset = cache.generate_with_cache(
        generator_llm=generator_llm,
        generator_embeddings=generator_embeddings,
        docs=[],  # Your docs here
        testset_size=20,
        llm_name="gpt-3.5-turbo"
    )
    
    logger.success(f"Test dataset ready: {len(test_dataset)} samples")
    
    # Step 3: Run retriever evaluation
    # results = evaluate_retrievers(test_dataset)
    
    return test_dataset


# ============================================================================
# UTILITY: Convert Cached Dataset to Pandas
# ============================================================================

def load_cached_dataset_as_df(cache_dir: str = "./testset_cache"):
    """Load and inspect cached datasets as pandas DataFrames"""
    import pandas as pd
    
    cache = TestDatasetCache(cache_dir=cache_dir)
    
    if not cache.metadata:
        logger.info("No cached datasets found")
        return None
    
    logger.info("\nAvailable Cached Datasets:")
    for i, (cache_key, info) in enumerate(cache.metadata.items(), 1):
        logger.info(f"{i}. {cache_key[:12]}... (created: {info['created_at']})")
    
    # Load first dataset as example
    first_key = list(cache.metadata.keys())[0]
    dataset = cache._load_dataset(first_key)
    
    if hasattr(dataset, 'to_pandas'):
        df = dataset.to_pandas()
        logger.info(f"\nDataset Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info("\nFirst few rows:")
        logger.info(f"\n{df.head()}")
        return df
    else:
        logger.warning("Dataset format not compatible with pandas")
        return None


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    RECOMMENDED USAGE:
    
    Replace your original generate_test_dataset() with:
    generate_test_dataset_cached()
    
    Benefits:
    - First run: Generates dataset (takes time)
    - Subsequent runs: Instant load from cache
    - Automatic cache invalidation if docs change
    - Separate caches for different LLMs/sizes
    """
    
    # Configure loguru logger
    logger.add("testset_cache.log", rotation="10 MB", level="INFO")
    
    logger.info(__doc__)
    logger.info("\nQuick Start:")
    logger.info("-" * 80)
    logger.info("""
    # Before (slow - regenerates every time):
    dataset = generate_test_dataset(llm, embeddings, docs, 10)
    
    # After (fast - uses cache):
    dataset = generate_test_dataset_cached(llm, embeddings, docs, 10)
    
    # Force regenerate:
    dataset = generate_test_dataset_cached(
        llm, embeddings, docs, 10, 
        force_regenerate=True
    )
    
    # Manage cache:
    cache = TestDatasetCache()
    cache.list_cached_datasets()
    cache.clear_cache()  # Clear all
    """)