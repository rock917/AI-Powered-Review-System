# app/utils/cache.py
# FeedbackIQ — SQLite Inference Cache
#
# PURPOSE:
# Avoid re-running DistilBERT on reviews that were already analyzed.
# Cache maps SHA256(review_text) → prediction result.
#
# SCHEMA:
# Table: predictions
#   hash          TEXT PRIMARY KEY   — SHA256 of review text
#   sentiment     TEXT               — 'Positive' or 'Negative'
#   confidence    REAL               — float 0-1
#   prob_positive REAL               — float 0-1
#   prob_negative REAL               — float 0-1
#   created_at    TEXT               — ISO timestamp

import os
import sqlite3
import hashlib
import json
from datetime import datetime

# ── DB location ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DB_PATH = os.path.join(PROJECT_DIR, 'feedbackiq_cache.db')


def _get_connection():
    """Get a SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_cache():
    """
    Create the predictions table if it does not exist.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            hash          TEXT PRIMARY KEY,
            sentiment     TEXT NOT NULL,
            confidence    REAL NOT NULL,
            prob_positive REAL NOT NULL,
            prob_negative REAL NOT NULL,
            created_at    TEXT NOT NULL
        )
    """)
    # Index on hash for fast lookups (already PK but explicit is clearer)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hash
        ON predictions(hash)
    """)
    conn.commit()
    conn.close()


def _hash_text(text: str) -> str:
    """
    Convert review text to a SHA256 hash.
    Same text always produces the same 64-char hex string.
    """
    return hashlib.sha256(
        text.strip().lower().encode('utf-8')
    ).hexdigest()


def get_cached(text: str) -> dict | None:
    """
    Look up a review in the cache.
    Returns result dict if found, None if not cached.
    """
    if not text or not text.strip():
        return None

    text_hash = _hash_text(text)

    try:
        conn  = _get_connection()
        row   = conn.execute(
            "SELECT * FROM predictions WHERE hash = ?",
            (text_hash,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        return {
            'sentiment'    : row['sentiment'],
            'confidence'   : row['confidence'],
            'prob_positive': row['prob_positive'],
            'prob_negative': row['prob_negative'],
            'truncated'    : False,
            'word_count'   : len(text.split()),
            'error'        : None,
            'from_cache'   : True
        }

    except Exception:
        return None


def store_result(text: str, result: dict):
    """
    Store a prediction result in the cache.
    Silently ignores errors — cache failure should never
    break the main prediction pipeline.
    """
    if not text or not text.strip():
        return
    if result.get('error'):
        return

    text_hash = _hash_text(text)

    try:
        conn = _get_connection()
        conn.execute("""
            INSERT OR IGNORE INTO predictions
            (hash, sentiment, confidence, prob_positive,
             prob_negative, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            text_hash,
            result['sentiment'],
            result['confidence'],
            result['prob_positive'],
            result['prob_negative'],
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_cache_stats() -> dict:
    """
    Return cache statistics for display in the UI.
    """
    try:
        conn  = _get_connection()
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        pos   = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE sentiment='Positive'"
        ).fetchone()[0]
        neg   = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE sentiment='Negative'"
        ).fetchone()[0]
        conn.close()
        return {
            'total'   : total,
            'positive': pos,
            'negative': neg,
            'db_path' : DB_PATH,
            'db_size' : os.path.getsize(DB_PATH) / 1024
                        if os.path.exists(DB_PATH) else 0
        }
    except Exception:
        return {'total': 0, 'positive': 0, 'negative': 0,
                'db_path': DB_PATH, 'db_size': 0}


def clear_cache():
    """Clear all cached predictions."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# Initialize on import
initialize_cache()