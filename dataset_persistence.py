"""
Dataset Loading with Persistence
Saves processed datasets to avoid repeated API calls
"""

import os
import pickle
import json
import hashlib
from pathlib import Path
from typing import List, Any, Optional
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from loguru import logger


# ============================================================================
# OPTION 1: PICKLE PERSISTENCE
# ============================================================================

def load_dataset_with_cache(
    filepath: str, 
    metadata_columns: List[str],
    cache_dir: str = "./cache",
    force_reload: bool = False
) -> List[Document]:
    """
    Load dataset with caching to avoid repeated processing.
    
    Args:
        filepath: Path to CSV file
        metadata_columns: Columns to store as metadata
        cache_dir: Directory to store cached datasets
        force_reload: If True, ignore cache and reload from CSV
    
    Returns:
        List of Document objects
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on file content and metadata columns
    cache_key = _generate_cache_key(filepath, metadata_columns)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Try to load from cache
    if not force_reload and os.path.exists(cache_file):
        logger.info(f"Loading dataset from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
            logger.info(f"Loaded {len(dataset)} documents from cache")
            return dataset
        except Exception as e:
            logger.error(f"Cache load failed: {e}. Reloading from CSV...")
    
    # Load from CSV
    logger.info(f"Loading dataset from CSV: {filepath}")
    dataset = _load_and_process_dataset(filepath, metadata_columns)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved {len(dataset)} documents to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Warning: Could not save cache: {e}")
    
    return dataset


def _generate_cache_key(filepath: str, metadata_columns: List[str]) -> str:
    """Generate a unique cache key based on file and configuration"""
    # Get file modification time and size
    file_stat = os.stat(filepath)
    file_info = f"{filepath}_{file_stat.st_mtime}_{file_stat.st_size}"
    
    # Include metadata columns in hash
    metadata_info = "_".join(sorted(metadata_columns))
    
    # Create hash
    cache_key = hashlib.md5(f"{file_info}_{metadata_info}".encode()).hexdigest()
    return cache_key


def _load_and_process_dataset(filepath: str, metadata_columns: List[str]) -> List[Document]:
    """Load and process the dataset (original logic)"""
    loader = CSVLoader(
        file_path=filepath,
        metadata_columns=metadata_columns,
    )

    dataset = loader.load()
    for doc in dataset:
        # Create a much more detailed document with all fields
        doc.page_content = f"""
Project Information:
------------------
Project Title: {doc.metadata.get('Project Title', 'N/A')}
Project Name: {doc.metadata.get('Project Name', 'N/A')}

Domain Classification:
---------------------
Primary Domain: {doc.metadata.get('Project Domain', 'N/A')}
Secondary Domain: {doc.metadata.get('Secondary Domain', 'N/A')}

Project Description:
-------------------
{doc.metadata.get('Description', 'N/A')}

Evaluation and Scoring:
----------------------
Judge Comments: {doc.metadata.get('Judge Comments', 'N/A')}
Overall Score: {doc.metadata.get('Score', 'N/A')} out of 100
Judge Individual Score: {doc.metadata.get('Judge Score', 'N/A')} out of 10

This project was evaluated based on multiple criteria including technical depth,
innovation, presentation quality, implementation completeness, and potential impact.
The evaluation considered both the technical execution and the real-world applicability
of the proposed solution.
""".strip()
    
    return dataset


# ============================================================================
# OPTION 2: JSON PERSISTENCE
# ============================================================================

def load_dataset_with_json_cache(
    filepath: str, 
    metadata_columns: List[str],
    cache_dir: str = "./cache",
    force_reload: bool = False
) -> List[Document]:
    """
    Load dataset with JSON caching (more portable than pickle)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = _generate_cache_key(filepath, metadata_columns)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Try to load from cache
    if not force_reload and os.path.exists(cache_file):
        logger.info(f"Loading dataset from JSON cache: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct Document objects
            dataset = [
                Document(
                    page_content=item['page_content'],
                    metadata=item['metadata']
                )
                for item in data
            ]
            logger.info(f" Loaded {len(dataset)} documents from JSON cache")
            return dataset
        except Exception as e:
            logger.error(f"JSON cache load failed: {e}. Reloading from CSV...")
    
    # Load from CSV
    logger.info(f"Loading dataset from CSV: {filepath}")
    dataset = _load_and_process_dataset(filepath, metadata_columns)
    
    # Save to JSON cache
    try:
        data = [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in dataset
        ]
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f" Saved {len(dataset)} documents to JSON cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Warning: Could not save JSON cache: {e}")
    
    return dataset


# ============================================================================
# OPTION 3: SMART CACHING WITH AUTO-INVALIDATION
# ============================================================================

class DatasetCache:
    """
    Smart dataset cache with automatic invalidation
    """
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
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
    
    def _get_cache_info(self, filepath: str, metadata_columns: List[str]) -> dict:
        """Get current file info"""
        file_stat = os.stat(filepath)
        return {
            'filepath': filepath,
            'mtime': file_stat.st_mtime,
            'size': file_stat.st_size,
            'metadata_columns': sorted(metadata_columns)
        }
    
    def _is_cache_valid(self, cache_key: str, current_info: dict) -> bool:
        """Check if cache is still valid"""
        if cache_key not in self.metadata:
            return False
        
        cached_info = self.metadata[cache_key]
        return (
            cached_info['filepath'] == current_info['filepath'] and
            cached_info['mtime'] == current_info['mtime'] and
            cached_info['size'] == current_info['size'] and
            cached_info['metadata_columns'] == current_info['metadata_columns']
        )
    
    def load(
        self, 
        filepath: str, 
        metadata_columns: List[str],
        force_reload: bool = False
    ) -> List[Document]:
        """Load dataset with smart caching"""
        current_info = self._get_cache_info(filepath, metadata_columns)
        cache_key = hashlib.md5(
            f"{filepath}_{'_'.join(sorted(metadata_columns))}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check if cache is valid
        if not force_reload and cache_file.exists() and self._is_cache_valid(cache_key, current_info):
            logger.info(f"Loading from valid cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load from source
        logger.info(f"Loading from CSV: {filepath}")
        dataset = _load_and_process_dataset(filepath, metadata_columns)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Update metadata
        self.metadata[cache_key] = current_info
        self._save_metadata()
        
        logger.info(f" Cached {len(dataset)} documents")
        return dataset
    
    def clear_cache(self):
        """Clear all cached files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Cache cleared")
    
    def list_cache(self):
        """List all cached datasets"""
        logger.info("\nCached Datasets:")
        logger.info("-" * 80)
        for cache_key, info in self.metadata.items():
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                logger.info(f"File: {info['filepath']}")
                logger.info(f"  Cache: {cache_key[:8]}... ({size_mb:.2f} MB)")
                logger.info(f"  Columns: {', '.join(info['metadata_columns'])}")
                logger.info()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Examples of how to use each caching method"""
    
    metadata_columns = [
        'Project Title', 'Project Name', 'Project Domain', 
        'Secondary Domain', 'Description', 'Judge Comments', 
        'Score', 'Judge Score'
    ]
    
    # OPTION 1: Simple pickle cache 
    logger.info("=" * 80)
    logger.info("OPTION 1: Simple Pickle Cache")
    logger.info("=" * 80)
    dataset1 = load_dataset_with_cache(
        filepath="your_data.csv",
        metadata_columns=metadata_columns,
        cache_dir="./cache",
        force_reload=False
    )
    logger.info(f"Loaded {len(dataset1)} documents\n")
    
    
    # OPTION 2: JSON cache (more portable)
    logger.info("=" * 80)
    logger.info("OPTION 2: JSON Cache (Portable)")
    logger.info("=" * 80)
    dataset2 = load_dataset_with_json_cache(
        filepath="your_data.csv",
        metadata_columns=metadata_columns,
        cache_dir="./cache_json",
        force_reload=False
    )
    logger.info(f"Loaded {len(dataset2)} documents\n")
    
    
    # OPTION 3: Smart cache with auto-invalidation
    logger.info("=" * 80)
    logger.info("OPTION 3: Smart Cache")
    logger.info("=" * 80)
    cache = DatasetCache(cache_dir="./smart_cache")
    
    # First load - reads from CSV
    dataset3 = cache.load(
        filepath="your_data.csv",
        metadata_columns=metadata_columns
    )
    
    # Second load - reads from cache
    dataset3 = cache.load(
        filepath="your_data.csv",
        metadata_columns=metadata_columns
    )
    
    # List cached datasets
    cache.list_cache()
    
    # Clear cache if needed
    # cache.clear_cache()


# ============================================================================
# INTEGRATION WITH YOUR RETRIEVER EVALUATION
# ============================================================================

def integrated_example():
    """How to integrate with your retriever evaluation code"""
    
    # Initialize cache
    cache = DatasetCache(cache_dir="./retriever_cache")
    
    # Load dataset (will cache automatically)
    metadata_columns = [
        'Project Title', 'Project Name', 'Project Domain', 
        'Secondary Domain', 'Description', 'Judge Comments', 
        'Score', 'Judge Score'
    ]
    
    documents = cache.load(
        filepath="projects.csv",
        metadata_columns=metadata_columns
    )
    
    logger.info(f"\n Ready to evaluate with {len(documents)} documents")
    
    # Convert to Document objects if needed
    from langchain_core.documents import Document
    
    # Your documents are already Document objects, ready to use!
    logger.info(f"First document preview:\n{documents[0].page_content[:200]}...")
    
    return documents


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_all_caches():
    """Clear all cache directories"""
    cache_dirs = ["./cache", "./cache_json", "./smart_cache", "./retriever_cache"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f" Cleared {cache_dir}")


def compare_cache_sizes():
    """Compare sizes of different cache formats"""
    import sys
    
    # Sample dataset
    sample_docs = [
        Document(
            page_content="This is a test document " * 100,
            metadata={"id": i, "title": f"Doc {i}"}
        )
        for i in range(100)
    ]
    
    # Pickle size
    pickle_data = pickle.dumps(sample_docs)
    pickle_size = sys.getsizeof(pickle_data)
    
    # JSON size
    json_data = json.dumps([
        {'page_content': doc.page_content, 'metadata': doc.metadata}
        for doc in sample_docs
    ])
    json_size = sys.getsizeof(json_data)
    
    logger.info("\nCache Format Comparison (100 documents):")
    logger.info("-" * 50)
    logger.info(f"Pickle: {pickle_size / 1024:.2f} KB")
    logger.info(f"JSON:   {json_size / 1024:.2f} KB")
    logger.info(f"Ratio:  {json_size / pickle_size:.2f}x")


# ============================================================================
# RECOMMENDED USAGE
# ============================================================================

if __name__ == "__main__":
    """
    RECOMMENDED: Use DatasetCache for best experience
    
    Features:
    - Automatic cache invalidation when CSV changes
    - Cache metadata tracking
    - Easy cache management
    - Fast pickle serialization
    """
    
    # Simple usage
    cache = DatasetCache()

    filepath = "./data/Projects_with_Domains.csv"
    
    dataset = cache.load(
        filepath=filepath,
        metadata_columns=['col1', 'col2', 'col3']
    )
    
    logger.info(f"\n Loaded {len(dataset)} documents")
    
    # Second call will use cache
    dataset = cache.load(
        filepath=filepath,
        metadata_columns=['col1', 'col2', 'col3']
    )