# ============================================================
# src/inference.py
# AI-Powered Customer Feedback Intelligence System
#
# PURPOSE:
# Provides a clean inference interface for sentiment prediction.
# Loads the fine-tuned DistilBERT model from Hugging Face once
# and reuses it for all predictions.
# ============================================================

import torch
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────
MAX_LENGTH = 256
BATCH_SIZE = 64

# Hugging Face model repository
MODEL_PATH = "TheRock45/feedbackiq-sentiment"

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Global model cache ───────────────────────────────────────
# The model is downloaded from Hugging Face only once and
# then kept in memory for subsequent predictions.
_tokenizer = None
_model = None


def load_model():
    """
    Load tokenizer and model from Hugging Face.
    Uses global cache so the model is only loaded once.
    """
    global _tokenizer, _model

    if _tokenizer is None or _model is None:

        _tokenizer = DistilBertTokenizerFast.from_pretrained(
            MODEL_PATH
        )

        _model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        _model = _model.to(device)
        _model.eval()

    return _tokenizer, _model


def predict_single(text: str) -> dict:
    """
    Predict sentiment for a single review text.

    Args:
        text (str): Raw review text

    Returns:
        dict with keys:
            sentiment      : 'Positive' or 'Negative'
            confidence     : float, probability of predicted class
            prob_positive  : float, probability of Positive
            prob_negative  : float, probability of Negative
            error          : str or None
    """

    # ── Input validation ─────────────────────────────────────
    if not isinstance(text, str):
        return _error_result("Input must be a string.")

    text = text.strip()

    if len(text) == 0:
        return _error_result("Review text is empty.")

    if len(text.split()) < 2:
        return _error_result(
            "Review is too short. Please enter at least 2 words."
        )

    # Warn if text is very long
    word_count = len(text.split())
    truncated = word_count > 180

    # ── Inference ────────────────────────────────────────────
    try:
        tokenizer, model = load_model()

        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()[0]

        prob_negative = float(probs[0])
        prob_positive = float(probs[1])

        predicted_idx = int(np.argmax(probs))

        sentiment = (
            'Positive'
            if predicted_idx == 1
            else 'Negative'
        )

        confidence = float(np.max(probs))

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'truncated': truncated,
            'word_count': word_count,
            'error': None
        }

    except Exception as e:
        return _error_result(
            f"Prediction failed: {str(e)}"
        )


def predict_batch(
    texts: list,
    show_progress: bool = True
) -> list:
    """
    Predict sentiment for a list of review texts.

    Uses batched inference for better performance.
    """

    if not texts:
        return []

    tokenizer, model = load_model()

    # ── Batch dataset ────────────────────────────────────────
    class SimpleDataset(Dataset):

        def __init__(self, texts, tokenizer, max_length):
            self.texts = [
                str(t) if t and str(t).strip()
                else "empty review"
                for t in texts
            ]

            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):

            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids':
                    enc['input_ids'].squeeze(0),

                'attention_mask':
                    enc['attention_mask'].squeeze(0)
            }

    dataset = SimpleDataset(
        texts,
        tokenizer,
        MAX_LENGTH
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_probs = []

    iterator = (
        tqdm(
            loader,
            desc="Analyzing reviews"
        )
        if show_progress
        else loader
    )

    with torch.no_grad():

        for batch in iterator:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()

            all_probs.extend(probs)

    # ── Build results ────────────────────────────────────────
    results = []

    for i, (text, probs) in enumerate(
        zip(texts, all_probs)
    ):

        prob_negative = float(probs[0])
        prob_positive = float(probs[1])

        predicted_idx = int(np.argmax(probs))

        sentiment = (
            'Positive'
            if predicted_idx == 1
            else 'Negative'
        )

        confidence = float(np.max(probs))

        word_count = (
            len(str(text).split())
            if text
            else 0
        )

        results.append({
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'truncated': word_count > 180,
            'word_count': word_count,
            'error': None
        })

    return results


def predict_dataframe(
    df: pd.DataFrame,
    text_column: str,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Run batch prediction on a pandas DataFrame.
    Adds prediction columns to the dataframe.
    """

    if text_column not in df.columns:

        raise ValueError(
            f"Column '{text_column}' not found in DataFrame.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    texts = df[text_column].fillna('').tolist()

    results = predict_batch(
        texts,
        show_progress=show_progress
    )

    df = df.copy()

    df['predicted_sentiment'] = [
        r['sentiment']
        for r in results
    ]

    df['confidence'] = [
        r['confidence']
        for r in results
    ]

    df['prob_positive'] = [
        r['prob_positive']
        for r in results
    ]

    df['prob_negative'] = [
        r['prob_negative']
        for r in results
    ]

    return df


def get_model_info() -> dict:
    """
    Return basic model information for display in the app.
    """

    return {
        'model_name':
            'distilbert-base-uncased (fine-tuned)',

        'test_accuracy':
            0.9501,

        'test_f1':
            0.9502,

        'test_precision':
            0.9487,

        'test_recall':
            0.9516,

        'baseline_accuracy':
            0.9212,

        'baseline_f1':
            0.9207,

        'training_samples':
            88001,

        'test_samples':
            11000,

        'max_length':
            MAX_LENGTH,

        'device':
            str(device)
    }


def _error_result(message: str) -> dict:
    """Return a standardized error result dict."""

    return {
        'sentiment': None,
        'confidence': None,
        'prob_positive': None,
        'prob_negative': None,
        'truncated': False,
        'word_count': 0,
        'error': message
    }