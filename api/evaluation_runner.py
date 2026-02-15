"""
Evaluation runner that supports streaming logs to UI.
Wraps evaluate_retrievers logic with log callbacks.
"""
import os
import time
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
from queue import Queue, Empty
import threading

# Import after setting up env
from dotenv import load_dotenv
load_dotenv()

import config  # loads API keys from config.py (edit there only)


def _emit_log(log_queue: Optional[Queue], message: str, level: str = "info") -> None:
    """Emit a log entry to the queue for UI streaming."""
    if log_queue is None:
        return
    ts = datetime.now().strftime("%I:%M:%S %p")
    log_queue.put({"timestamp": ts, "message": message, "level": level})


def run_evaluation(
    log_queue: Optional[Queue] = None,
    file_path: str = "./data/Projects_with_Domains.csv",
) -> Dict[str, Any]:
    """
    Run the full retriever evaluation pipeline.
    If log_queue is provided, emits timestamped log entries for UI streaming.
    Returns dict with results_summary, execution_log, executive_summary.
    """
    # Lazy imports to avoid loading heavy deps before API starts
    from evaluate_retrievers import (
        load_dataset,
        generate_test_dataset,
        qdrant_vector_store,
        chat_prompt_template,
        generate_chat_model,
        naive_retriever_chain,
        bm25_retriever_chain,
        contextual_compression_retriever_chain,
        multiquery_retriever_chain,
        parent_document_retriever_chain,
        ensemble_retriever_chain,
        create_evaluation_dataset,
        evaluate_ragas_dataset,
        get_langsmith_cost_stats,
        EVALUATION_SESSION_ID,
    )
    from langsmith import Client, tracing_context
    from langchain_community.retrievers import BM25Retriever
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    import pandas as pd

    metadata_cols = [
        "Project Title", "Project Name", "Project Domain",
        "Secondary Domain", "Description", "Judge Comments",
        "Score", "Judge Score"
    ]
    path = file_path or "./data/Projects_with_Domains.csv"

    execution_log: List[Dict[str, str]] = []
    results_summary: List[Dict[str, Any]] = []

    def log(msg: str, level: str = "info"):
        ts = datetime.now().strftime("%I:%M:%S %p")
        entry = {"timestamp": ts, "message": msg, "level": level}
        execution_log.append(entry)
        if log_queue:
            log_queue.put(entry)

    try:
        log("Initializing advanced retrieval evaluation...")
        log("Loading CSV dataset...")
        dataset = load_dataset(path, metadata_cols)
        log(f"Loaded {len(dataset)} documents from dataset")

        log("Creating test dataset generator (Ragas)...")
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1", temperature=0.2, seed=42))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        log("Generating test dataset (this may take a minute)...")
        ds = generate_test_dataset(generator_llm, generator_embeddings, dataset)
        df = ds.to_pandas()
        log(f"Test dataset ready: {len(df)} question-answer pairs")

        log("Creating Qdrant vector store...")
        vector_store = qdrant_vector_store(dataset)
        log("Vector store created successfully")

        log("Setting up RAG prompt template and chat model...")
        rag_prompt = chat_prompt_template()
        chat_model = generate_chat_model()

        log("Building retriever chains...")
        naive = naive_retriever_chain(rag_prompt, chat_model, vector_store)
        log("  ✓ Naive retriever")
        bm25 = bm25_retriever_chain(rag_prompt, chat_model, dataset)
        log("  ✓ BM25 retriever")
        ctx_comp = contextual_compression_retriever_chain(rag_prompt, chat_model, naive)
        log("  ✓ Contextual compression retriever")
        multi_q = multiquery_retriever_chain(rag_prompt, chat_model, naive)
        log("  ✓ Multi-query retriever")
        parent_doc = parent_document_retriever_chain(rag_prompt, chat_model, dataset)
        log("  ✓ Parent document retriever")
        ensemble = ensemble_retriever_chain(rag_prompt, chat_model, [naive, bm25, ctx_comp, multi_q, parent_doc])
        log("  ✓ Ensemble retriever")

        retriever_map = {
            "naive_retriever": naive,
            "bm25_retriever": bm25,
            "contextual_compression_retriever": ctx_comp,
            "multi_query_retriever": multi_q,
            "parent_document_retriever": parent_doc,
            "ensemble_retriever": ensemble,
        }

        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
        project_name = os.environ.get("LANGCHAIN_PROJECT", "Advanced-Retrieval-Evaluation")

        for retriever_name in retriever_map:
            log(f"Evaluating: {retriever_name}...", "info")
            client = Client()
            project_retriever_name = f"{project_name}-{retriever_name}"
            os.environ["LANGCHAIN_PROJECT"] = project_retriever_name

            start_time = time.time()
            with tracing_context(project_name=project_retriever_name, client=client):
                log(f"  Generating evaluation dataset for {retriever_name}...")
                eval_ds = create_evaluation_dataset(
                    ds, retriever_map[retriever_name], rag_prompt, chat_model,
                    retriever_name=retriever_name, session_id=EVALUATION_SESSION_ID,
                )
                log(f"  Running RAGAS evaluation for {retriever_name}...")
                eval_results = evaluate_ragas_dataset(eval_ds, evaluator_llm, project_retriever_name)

            latency = time.time() - start_time
            results_df = eval_results.to_pandas()
            numeric_cols = results_df.select_dtypes(include=["number"]).columns
            results_dict = results_df[numeric_cols].mean().to_dict()

            result_row = {
                "retriever": retriever_name,
                "context_recall": round(results_dict.get("context_recall", 0), 4),
                "context_entity_recall": round(results_dict.get("context_entity_recall", 0), 4),
                "noise_sensitivity": round(results_dict.get("noise_sensitivity_relevant", 0), 4),
                "latency_seconds": round(latency, 2),
                "langsmith_total_cost_usd": None,
                "langsmith_total_tokens": None,
                "langsmith_llm_calls": None,
                "langsmith_total_latency": None,
            }
            results_summary.append(result_row)
            if log_queue:
                log_queue.put({"_event": "retriever_result", "result": result_row})
            log(f"  ✓ {retriever_name} complete: recall={results_dict.get('context_recall', 0):.2f}, latency={latency:.1f}s")

        log("Fetching cost data from LangSmith...")
        try:
            client = Client()
            for result in results_summary:
                rn = result["retriever"]
                try:
                    cost_stats = get_langsmith_cost_stats(
                        client,
                        f"{project_name}-{rn}",
                        retriever_name=rn,
                        session_id=EVALUATION_SESSION_ID,
                    )
                    result["langsmith_total_cost_usd"] = cost_stats["total_cost"]
                    result["langsmith_total_tokens"] = cost_stats["total_tokens"]
                    result["langsmith_llm_calls"] = cost_stats["run_count"]
                    result["langsmith_total_latency"] = cost_stats["total_latency_seconds"]
                except Exception:
                    pass
        except Exception as e:
            log(f"Could not fetch LangSmith costs: {e}", "warning")

        log("Saving results to CSV...")
        summary_df = pd.DataFrame(results_summary)
        os.makedirs("./data", exist_ok=True)
        summary_df.to_csv("./data/retriever_evaluation_results.csv", index=False)
        log("Evaluation complete!")

        executive_summary = _build_executive_summary(results_summary)
        return {
            "success": True,
            "results_summary": results_summary,
            "execution_log": execution_log,
            "executive_summary": executive_summary,
            "session_id": EVALUATION_SESSION_ID,
        }

    except Exception as e:
        err_str = str(e)
        log(f"Error: {err_str}", "error")
        # User-friendly message for OpenAI quota/billing errors
        try:
            from openai import RateLimitError
            if isinstance(e, RateLimitError):
                err_str = "OpenAI quota exceeded. Check: platform.openai.com/usage | Add billing: platform.openai.com/account/billing"
        except ImportError:
            if "429" in err_str or "insufficient_quota" in err_str.lower():
                err_str = "OpenAI quota exceeded. Check: platform.openai.com/usage | Add billing: platform.openai.com/account/billing"
        import traceback
        log(traceback.format_exc(), "error")
        return {
            "success": False,
            "error": err_str,
            "results_summary": results_summary,
            "execution_log": execution_log,
            "executive_summary": None,
            "session_id": os.environ.get("LANGCHAIN_PROJECT", "unknown").split("-")[-1][:8] if "LANGCHAIN_PROJECT" in os.environ else "unknown",
        }


def _build_executive_summary(results_summary: List[Dict[str, Any]]) -> str:
    """Generate executive summary from evaluation results."""
    if not results_summary:
        return "No evaluation results available. Run the evaluation to generate a summary."

    def _recall(r):
        cr = r.get("context_recall", 0) or 0
        er = r.get("context_entity_recall", 0) or 0
        return (float(cr) + float(er)) / 2

    best = max(results_summary, key=_recall)
    worst = min(results_summary, key=_recall)
    cr = best.get("context_recall", 0) or 0
    er = best.get("context_entity_recall", 0) or 0

    parts = [
        f"This report summarizes the evaluation of {len(results_summary)} advanced retrieval strategies "
        f"on the Projects dataset using RAGAS metrics (Context Recall, Entity Recall, Noise Sensitivity). "
    ]
    parts.append(
        f"The top-performing retriever is {best['retriever']} with context recall {float(cr):.2%} "
        f"and entity recall {float(er):.2%}."
    )
    if best["retriever"] != worst["retriever"]:
        parts.append(
            f"The {worst['retriever']} showed the lowest combined recall in this run."
        )
    parts.append(
        "Advanced strategies (ensemble, contextual compression, multi-query) typically improve retrieval quality "
        "but may increase latency and cost. Review the detailed metrics below to choose the best retriever for your use case."
    )
    return " ".join(parts)
