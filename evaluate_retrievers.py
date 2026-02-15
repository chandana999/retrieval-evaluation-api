"""Evaluate various retrievers"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
import pandas as pd
from uuid import uuid4
from operator import itemgetter
from dotenv import load_dotenv
load_dotenv()

import config  # loads API keys from config.py (edit there only)

from dataset_persistence import DatasetCache
from testset_persistence import TestDatasetCache, generate_test_dataset_cached

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import(
    ParentDocumentRetriever,
    MultiQueryRetriever,
    EnsembleRetriever,
)
from langsmith import Client, tracing_context
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.storage import InMemoryStore
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient, models

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import(
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness, 
    ResponseRelevancy, 
    ContextEntityRecall, 
    NoiseSensitivity
)


# Create unique session ID for this evaluation run
EVALUATION_SESSION_ID = uuid4().hex[:8]
os.environ["LANGCHAIN_PROJECT"] = f"Advanced-Retrieval-Eval-{EVALUATION_SESSION_ID}"


file_path=f"./data/Projects_with_Domains.csv"
metadata_columns = [
     "Project Title",
      "Project Domain",
      "Secondary Domain",
      "Description",
      "Judge Comments",
      "Score",
      "Project Name",
      "Judge Score",
]

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def load_dataset(filepath: str, metadata_columns: List[str]) -> List[Any]:
    """   
    Combine multiple fields to create longer documents for Ragas.
    Ragas requires documents with at least 100 tokens.
    """
    cache = DatasetCache(cache_dir="./dataset_cache")

    dataset = cache.load(
        filepath=filepath,
        metadata_columns=[
            'Project Title', 'Project Name', 'Project Domain',
            'Secondary Domain', 'Description', 'Judge Comments',
            'Score', 'Judge Score'
        ]
    )
    
    return dataset

def generate_test_dataset(
    generator_llm: LangchainLLMWrapper, 
    generator_embeddings: LangchainEmbeddingsWrapper,
    docs: List[Any],
    testset_size: int = 10) -> Dict[str, Any]:
    """Generate a test dataset using the generator LLM and embeddings"""
    cache = TestDatasetCache()
    dataset = generate_test_dataset_cached(
        generator_llm, generator_embeddings, docs, testset_size,
        force_regenerate=False
    )
    # generator = TestsetGenerator(
    #     llm=generator_llm,
    #     embedding_model=generator_embeddings,
    # )
    # dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
    return dataset

def qdrant_vector_store(docs: List[Any]) -> Qdrant:
    """Setup a Qdrant vector store"""
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")

    return Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="ragas_test_dataset",
    )

def chat_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt template"""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)

def generate_chat_model() -> ChatOpenAI:
    """Create a chat model with deterministic settings"""
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, seed=42)

def naive_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    vector_store: Qdrant,
    k: int = 10
) -> Dict[str, Any]: 
    """Create a naive retriever"""
    naive_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    return naive_retriever

def bm25_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    docs: List[Any],
) -> Dict[str, Any]:
    """Create a BM25 retriever"""
    bm25_retriever = BM25Retriever.from_documents(docs)
    
    return bm25_retriever

def contextual_compression_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    naive_retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a contextual compression retriever"""
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=naive_retriever
    )

    return compression_retriever

def multiquery_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    naive_retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a multi-query retriever"""
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=naive_retriever, llm=chat_model
    ) 

    return multi_query_retriever

def parent_document_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    docs: List[Any],
) -> Dict[str, Any]:
    """Create a parent document retriever"""
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
    client = QdrantClient(location=":memory:")

    client.create_collection(
        collection_name="parent_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

    parent_document_vectorstore = QdrantVectorStore(
        collection_name="parent_documents", 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), 
        client=client
    )

    store = InMemoryStore()

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore = parent_document_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    parent_document_retriever.add_documents(docs, ids=None)

    return parent_document_retriever

def ensemble_retriever_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI, 
    retriever_list: List[Any],
) -> Dict[str, Any]:
    """Create a ensemble retriever"""

    equal_weighting = [1/len(retriever_list)] * len(retriever_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=equal_weighting
    )

    return ensemble_retriever


def make_lcel_chain(
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI,
    retriever: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make a LCEL chain
    """
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )

def cast_text(response: AIMessage) -> str:
    """Cast a response to a string"""
    return response.content if isinstance(response, AIMessage) else response

def create_evaluation_dataset(
    dataset: List[Any], 
    retriever: Dict[str, Any], 
    rag_prompt: ChatPromptTemplate, 
    chat_model: ChatOpenAI,
    retriever_name: str = None,
    session_id: str = None,
) -> EvaluationDataset:
    """Create a evaluation dataset with session tracking"""
    logger.info(f"Make LCEL RAG chain for `{retriever_name}`")
    lcel_chain = make_lcel_chain(rag_prompt, chat_model, retriever)
    
    # Process each test case with metadata tracking
    for doc in dataset:
        user_input = getattr(doc.eval_sample, "user_input", None) or \
            getattr(doc.eval_sample, "question", None)
        if user_input:
            # Invoke with metadata in config (correct way for LangSmith)
            retrieved_docs = lcel_chain.invoke(
                {"question": user_input},
                config={
                    "metadata": {
                        "retriever": retriever_name,
                        "session_id": session_id or EVALUATION_SESSION_ID,
                        "task": "retrieval_evaluation"
                    },
                    "tags": [retriever_name, f"session-{session_id or EVALUATION_SESSION_ID}"]
                }
            )
            doc.eval_sample.response = cast_text(retrieved_docs["response"])
            doc.eval_sample.retrieved_contexts = [
                context.page_content for context in retrieved_docs["context"]
            ]
    
    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
    
    return evaluation_dataset

def evaluate_ragas_dataset(
    dataset: EvaluationDataset, 
    evaluator_llm: LangchainLLMWrapper, 
    project_name: str = None
) -> Dict[str, Any]:
    """Evaluate a RAGAS dataset with retriever-specific metrics
    
    Focus on retrieval quality metrics:
    - LLMContextRecall: Did we retrieve the reference documents?
    - ContextEntityRecall: Did we capture key entities?
    - NoiseSensitivity: Are we filtering irrelevant documents?
    
    Note: Generation metrics (Faithfulness, FactualCorrectness, ResponseRelevancy) 
    are excluded to reduce cost when comparing retrievers.
    """
    return evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),      # Primary retrieval metric
            ContextEntityRecall(),   # Entity coverage metric
            NoiseSensitivity(),      # Noise filtering metric
        ],
        llm=evaluator_llm,
        run_config=RunConfig(
            timeout=600,       # 10 min per job (avoids TimeoutError under load)
            max_workers=4,     # Fewer concurrent calls = less rate-limit/timeout pressure
            max_retries=5,
            max_wait=120,
        ),
        raise_exceptions=False,
    )

def get_langsmith_cost_stats(
    client: Client,
    project_name: str, 
    retriever_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    """Fetch pre-computed cost and latency statistics from LangSmith
    
    Args:
        project_name: LangSmith project name
        retriever_name: Optional filter for specific retriever runs (not used for project stats)
        session_id: Optional filter for specific evaluation session (not used for project stats)
    
    Returns:
        Dictionary with LangSmith's pre-computed statistics including p99, costs, tokens
    """
    filter_conditions = []
    
    if session_id:
        filter_conditions.append(f'has(tags, "session-{session_id}")')
    
    if retriever_name:
        filter_conditions.append(f'has(tags, "{retriever_name}")')
    
    filter_str = f'and({", ".join(filter_conditions)})' if filter_conditions else None
    
    # Get your tagged runs (the 10 question runs)
    runs = list(client.list_runs(
        project_name=project_name,
        run_type="chain",
        # filter='eq(name, "ragas evaluation")',
        # limit=100  # Get all matching runs
    ))
    
    logger.info(f"Found {len(runs)} tagged runs for {retriever_name}")
    
    if not runs:
        return {
            "total_cost": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "run_count": 0,
            "total_latency_seconds": 0,
            "avg_latency_seconds": 0,
        }
    
    # Step 2: Get unique trace_ids from your runs
    trace_ids = set(run.trace_id for run in runs)
    
    # Step 3: For each trace, find the root "ragas evaluation" run
    ragas_runs = []
    for trace_id in trace_ids:
        # logger.info(f"Processing trace_id: {trace_id}")
        trace_runs = list(client.list_runs(
            project_name=project_name,
            trace_id=trace_id,
            is_root=True
        ))
        ragas_runs.extend([r for r in trace_runs if r.name == "ragas evaluation" or r.name == "RunnableSequence"])
    
    logger.info(f"Found {len(ragas_runs)} ragas evaluation runs")
    # logger.info(ragas_runs)
    
    # Step 4: Aggregate costs/latency from ragas evaluation runs
    total_cost = sum(r.total_cost for r in ragas_runs if r.total_cost)
    total_tokens = sum(r.total_tokens for r in ragas_runs if r.total_tokens)
    prompt_tokens = sum(r.prompt_tokens for r in ragas_runs if r.prompt_tokens)
    completion_tokens = sum(r.completion_tokens for r in ragas_runs if r.completion_tokens)
    
    latencies = [(r.end_time - r.start_time).total_seconds() 
                 for r in ragas_runs if r.end_time and r.start_time]
    
    return {
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "run_count": len(ragas_runs),
        "total_latency_seconds": sum(latencies),
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0,
    }


def main():
    logger.info("Loading CSV dataset")
    dataset = load_dataset(file_path, metadata_columns)
    
    logger.info("Generate test dataset")
    # Use temperature=0 for deterministic test generation
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1", temperature=0.2, seed=42))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    ds = generate_test_dataset(generator_llm, generator_embeddings, dataset)
    # logger.info(ds.to_pandas()[:2])

    df = ds.to_pandas()
    logger.info(f"Size of test dataset - {len(df)}")
    # logger.info(df["synthesizer_name"].value_counts())

    logger.info("Create Qdrant vector store")
    vector_store = qdrant_vector_store(dataset)

    logger.info("Create rag prompt template")
    rag_prompt = chat_prompt_template()

    logger.info("Create chat model")
    chat_model = generate_chat_model()

    retriever_map = {
        "naive_retriever": naive_retriever_chain(rag_prompt, chat_model, vector_store),
        "bm25_retriever": bm25_retriever_chain(rag_prompt, chat_model, dataset),
        "contextual_compression_retriever": contextual_compression_retriever_chain(
            rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)
        ),
        "multi_query_retriever": multiquery_retriever_chain(rag_prompt, chat_model, 
        naive_retriever_chain(rag_prompt, chat_model, vector_store)),
        "parent_document_retriever": parent_document_retriever_chain(rag_prompt, chat_model, dataset),
        "ensemble_retriever": ensemble_retriever_chain(rag_prompt, chat_model, 
        [
            naive_retriever_chain(rag_prompt, chat_model, vector_store), 
            bm25_retriever_chain(rag_prompt, chat_model, dataset), 
            contextual_compression_retriever_chain(rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
            multiquery_retriever_chain(rag_prompt, chat_model, 
            naive_retriever_chain(rag_prompt, chat_model, vector_store)), 
            parent_document_retriever_chain(rag_prompt, chat_model, dataset)
        ]),
    }
    
    # Use temperature=0 for deterministic evaluation
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    
    # Store results for comparison
    project_name = os.environ.get("LANGCHAIN_PROJECT", "Advanced-Retrieval-Evaluation")
    results_summary = []
    c  = 0
    for retriever_name in retriever_map:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {retriever_name}")
        logger.info(f"{'='*60}")

        client = Client()
        # client.create_project()
        # Each retriever should have its own project to capture costs and latency by retriever
        project_retreiver_name = f"{project_name}-{retriever_name}"
        os.environ["LANGCHAIN_PROJECT"] = project_retreiver_name

        # logger.info(f"Set LANGCHAIN_PROJECT to: {os.environ['LANGCHAIN_PROJECT']}")
        logger.info(f"Expected project: {project_retreiver_name}")
        
        # Track timing
        start_time = time.time()

        logger.info(f"Generate evaluation dataset for `{retriever_name}`")
        # eval_ds = create_evaluation_dataset(
        #     ds,
        #     retriever_map[retriever_name],
        #     rag_prompt,
        #     chat_model,
        #     retriever_name=retriever_name,
        #     session_id=EVALUATION_SESSION_ID,
        # )

        with tracing_context(project_name=project_retreiver_name, client=client):
            eval_ds = create_evaluation_dataset(
                ds, retriever_map[retriever_name], rag_prompt, chat_model,
                    retriever_name=retriever_name, session_id=EVALUATION_SESSION_ID
                )

            logger.info(f"Evaluate `{retriever_name}` using RAGAS")
            eval_results = evaluate_ragas_dataset(eval_ds, evaluator_llm, project_retreiver_name)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Convert EvaluationResult to dict - only numeric columns
        results_df = eval_results.to_pandas()
        # Select only numeric columns for mean calculation
        numeric_cols = results_df.select_dtypes(include=['number']).columns
        results_dict = results_df[numeric_cols].mean().to_dict()
        
        # Store results
        results_summary.append({
            "retriever": retriever_name,
            "context_recall": results_dict.get("context_recall", 0),
            "context_entity_recall": results_dict.get("context_entity_recall", 0),
            "noise_sensitivity": results_dict.get("noise_sensitivity_relevant", 0),
            "latency_seconds": latency,
        })
        
        logger.info(f"Results: {results_dict}")
        logger.info(f"Latency: {latency:.2f}s")
        # if c >= 1:
        #     break
        # c += 1
    
    # Fetch cost data from LangSmith
    logger.info(f"\n{'='*60}")
    logger.info("FETCHING COST DATA FROM LANGSMITH")
    logger.info(f"{'='*60}")
    
    try:
        # Get overall project costs for THIS session only
        # overall_cost_stats = get_langsmith_cost_stats(
        #     project_name, 
        #     session_id=EVALUATION_SESSION_ID,
        # )
        # logger.info(f"\nOverall Session Stats (Session ID: {EVALUATION_SESSION_ID}):")
        # logger.info(f"  Total Cost: ${overall_cost_stats['total_cost']:.4f}")
        # logger.info(f"  Total Tokens: {overall_cost_stats['total_tokens']:,}")
        # logger.info(f"  Total Runs: {overall_cost_stats['run_count']}")
        
        # Try to get per-retriever costs for THIS session
        client = Client()
        logger.info(f"Session ID: {EVALUATION_SESSION_ID}")
        for result in results_summary:
            retriever_name = result["retriever"]
            project_retreiver_name = f"{project_name}-{retriever_name}"
            try:
                cost_stats = get_langsmith_cost_stats(
                    client,
                    project_retreiver_name, 
                    retriever_name=retriever_name,
                    session_id=EVALUATION_SESSION_ID,
                )
                logger.info(cost_stats)
                result["langsmith_total_cost_usd"] = cost_stats["total_cost"]
                result["langsmith_total_tokens"] = cost_stats["total_tokens"]
                result["langsmith_llm_calls"] = cost_stats["run_count"]
                result["langsmith_total_latency"] = cost_stats["total_latency_seconds"]
                
                logger.info(f"{retriever_name}:")
                logger.info(f"  Total Cost: ${cost_stats['total_cost']:.4f}")
                logger.info(f"  Tokens: {cost_stats['total_tokens']:,}")
            except Exception as e:
                logger.warning(f"Could not fetch stats for {retriever_name}: {e}")
                result["langsmith_total_cost_usd"] = None
                result["langsmith_total_tokens"] = None
                result["langsmith_llm_calls"] = None
                result["langsmith_total_latency"] = None
    except Exception as e:
        logger.warning(f"Could not fetch LangSmith costs: {e}")
        logger.info("Continuing without cost data...")
    
    # Print summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'='*60}")
    
    summary_df = pd.DataFrame(results_summary)
    # logger.info(f"\n{summary_df.to_string()}")
    
    # Save results
    summary_df.to_csv("./data/retriever_evaluation_results.csv", index=False)
    logger.info("\nResults saved to: ./data/retriever_evaluation_results.csv")
    logger.info("\nFor detailed cost analysis, check LangSmith dashboard at:")
    logger.info("https://smith.langchain.com/")
    

if __name__ == "__main__":
    main()
