# app/utils/validators.py
# FeedbackIQ — Input and File Validation

import pandas as pd

ACCEPTED_TEXT_COLUMNS = [
    # Amazon / Kaggle standard
    'reviewText', 'review_text', 'cleaned_text',
    # Generic
    'text', 'Text', 'review', 'Review',
    'comment', 'Comment', 'feedback', 'Feedback',
    # With spaces (Trustpilot, Google, etc.)
    'Review Text', 'review text', 'Review Body',
    'review body', 'Review Content', 'review content',
    'Customer Review', 'customer review',
    'Body', 'body', 'Content', 'content',
    'Description', 'description', 'Message', 'message',
]

MAX_ROWS = 10000
MIN_WORDS = 2


def validate_review_text(text: str) -> dict:
    """
    Validate a single review text input.
    Returns dict with 'valid' bool and 'error' message.
    """
    if not text or not isinstance(text, str):
        return {'valid': False, 'error': 'Please enter a review.'}

    text = text.strip()

    if len(text) == 0:
        return {'valid': False, 'error': 'Review text is empty.'}

    words = text.split()
    if len(words) < MIN_WORDS:
        return {'valid': False,
                'error': f'Review too short. Enter at least {MIN_WORDS} words.'}

    if len(text) > 10000:
        return {'valid': False,
                'error': 'Review too long. Maximum 10,000 characters.'}

    return {'valid': True, 'error': None}


def validate_csv(df: pd.DataFrame) -> dict:
    """
    Validate an uploaded CSV DataFrame.
    Returns dict with validation results.
    """
    if df is None or len(df) == 0:
        return {
            'valid'      : False,
            'error'      : 'Uploaded file is empty.',
            'text_column': None,
            'row_count'  : 0,
            'truncated'  : False
        }

    # Find text column
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

        # Detect optional rating column
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