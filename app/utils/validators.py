# app/utils/validators.py
# FeedbackIQ — Input and File Validation

import torch
import pandas as pd

# Automatically detect environment
# GPU available = running locally = higher limit
# CPU only = running on cloud = conservative limit
IS_GPU = torch.cuda.is_available()
MAX_ROWS = 10000 if IS_GPU else 500

ACCEPTED_TEXT_COLUMNS = [
    'reviewText', 'review_text', 'cleaned_text',
    'text', 'Text', 'review', 'Review',
    'comment', 'Comment', 'feedback', 'Feedback',
    'Review Text', 'review text', 'Review Body',
    'review body', 'Review Content', 'review content',
    'Customer Review', 'customer review',
    'Body', 'body', 'Content', 'content',
    'Description', 'description', 'Message', 'message',
]


def validate_review_text(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {'valid': False, 'error': 'Please enter a review.'}
    text = text.strip()
    if len(text) == 0:
        return {'valid': False, 'error': 'Review text is empty.'}
    if len(text.split()) < 2:
        return {'valid': False,
                'error': 'Review too short. Enter at least 2 words.'}
    if len(text) > 10000:
        return {'valid': False,
                'error': 'Review too long. Maximum 10,000 characters.'}
    return {'valid': True, 'error': None}


def validate_csv(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {
            'valid'      : False,
            'error'      : 'Uploaded file is empty.',
            'text_column': None,
            'row_count'  : 0,
            'truncated'  : False
        }

    text_col = None
    for col in ACCEPTED_TEXT_COLUMNS:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        return {
            'valid'      : False,
            'error'      : (
                f"No review text column found.\n"
                f"Your columns: {df.columns.tolist()}\n"
                f"Expected one of: {ACCEPTED_TEXT_COLUMNS}"
            ),
            'text_column': None,
            'row_count'  : len(df),
            'truncated'  : False
        }

    truncated = len(df) > MAX_ROWS

    rating_col = None
    for col in ['Rating', 'rating', 'Score', 'score', 'Stars', 'stars']:
        if col in df.columns:
            rating_col = col
            break

    return {
        'valid'      : True,
        'error'      : None,
        'text_column': text_col,
        'rating_col' : rating_col,
        'row_count'  : min(len(df), MAX_ROWS),
        'truncated'  : truncated,
        'columns'    : df.columns.tolist()
    }