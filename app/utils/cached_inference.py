# app/utils/cached_inference.py
# FeedbackIQ — Cache-Aware Batch Inference
#
# PURPOSE:
# Wraps the raw inference pipeline with cache lookup.
# For each review:
#   1. Check SQLite cache
#   2. If HIT  → return instantly
#   3. If MISS → run DistilBERT → store in cache → return

import pandas as pd
from utils.cache import get_cached, store_result, get_cache_stats


def cached_batch_predict(texts: list,
                          progress_callback=None) -> tuple[list, dict]:
    """
    Run batch prediction with SQLite caching.

    Args:
        texts            : list of review text strings
        progress_callback: optional function(current, total, status_msg)

    Returns:
        results : list of prediction dicts (same format as predict_batch)
        stats   : dict with cache hit/miss counts
    """
    from src.inference import predict_batch

    total     = len(texts)
    results   = [None] * total
    cache_hits= 0
    miss_idx  = []
    miss_texts= []

    # ── Pass 1: Cache lookup ──────────────────────────────────
    if progress_callback:
        progress_callback(0, total, "Checking cache...")

    for i, text in enumerate(texts):
        cached = get_cached(str(text) if text else '')
        if cached:
            results[i] = cached
            cache_hits += 1
        else:
            miss_idx.append(i)
            miss_texts.append(str(text) if text else '')

    cache_misses = len(miss_idx)

    # ── Pass 2: Run inference on cache misses only ────────────
    if miss_texts:
        if progress_callback:
            progress_callback(
                cache_hits, total,
                f"Cache: {cache_hits} hits · "
                f"Running model on {cache_misses} new reviews..."
            )

        # Run in sub-batches to report progress
        SUB_BATCH = 256
        inference_results = []

        for batch_start in range(0, len(miss_texts), SUB_BATCH):
            batch = miss_texts[batch_start: batch_start + SUB_BATCH]
            batch_results = predict_batch(batch, show_progress=False)
            inference_results.extend(batch_results)

            done = cache_hits + batch_start + len(batch)
            if progress_callback:
                progress_callback(
                    done, total,
                    f"Analyzing... {done:,} / {total:,}"
                )

        # Store new results in cache and fill in results list
        for idx, text, result in zip(miss_idx, miss_texts, inference_results):
            results[idx] = result
            store_result(text, result)

    if progress_callback:
        progress_callback(total, total, "Complete!")

    stats = {
        'total'       : total,
        'cache_hits'  : cache_hits,
        'cache_misses': cache_misses,
        'hit_rate'    : cache_hits / total * 100 if total > 0 else 0
    }

    return results, stats


def cached_predict_dataframe(df: pd.DataFrame,
                              text_column: str,
                              progress_callback=None) -> tuple[pd.DataFrame, dict]:
    """
    Run cached batch prediction on a DataFrame.
    Adds prediction columns and returns (df, cache_stats).
    """
    texts   = df[text_column].fillna('').tolist()
    results, stats = cached_batch_predict(texts, progress_callback)

    df = df.copy()
    df['predicted_sentiment'] = [r['sentiment']     for r in results]
    df['confidence']          = [r['confidence']    for r in results]
    df['prob_positive']       = [r['prob_positive'] for r in results]
    df['prob_negative']       = [r['prob_negative'] for r in results]

    return df, stats