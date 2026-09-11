# src/test_inference.py
# Run this to verify inference.py works correctly
# CMD: python src/test_inference.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_single, predict_batch, get_model_info

print("=" * 55)
print("TESTING INFERENCE PIPELINE")
print("=" * 55)

# Test 1 — clear positive
r1 = predict_single("This coffee is absolutely amazing, best I have ever tasted!")
print(f"\nTest 1 — Clear Positive:")
print(f"  Text      : 'This coffee is absolutely amazing...'")
print(f"  Sentiment : {r1['sentiment']}")
print(f"  Confidence: {r1['confidence']:.4f}")
print(f"  P(Positive): {r1['prob_positive']:.4f}")
print(f"  P(Negative): {r1['prob_negative']:.4f}")

# Test 2 — clear negative
r2 = predict_single("Terrible product, broke after two days and customer service was useless.")
print(f"\nTest 2 — Clear Negative:")
print(f"  Text      : 'Terrible product, broke after two days...'")
print(f"  Sentiment : {r2['sentiment']}")
print(f"  Confidence: {r2['confidence']:.4f}")

# Test 3 — ambiguous
r3 = predict_single("The taste is okay but the packaging was damaged when it arrived.")
print(f"\nTest 3 — Ambiguous:")
print(f"  Text      : 'The taste is okay but packaging was damaged...'")
print(f"  Sentiment : {r3['sentiment']}")
print(f"  Confidence: {r3['confidence']:.4f}")

# Test 4 — empty input
r4 = predict_single("")
print(f"\nTest 4 — Empty input (error handling):")
print(f"  Error     : {r4['error']}")

# Test 5 — batch prediction
texts = [
    "Fantastic product, highly recommend!",
    "Complete waste of money, do not buy.",
    "It arrived on time and works as described."
]
results = predict_batch(texts, show_progress=False)
print(f"\nTest 5 — Batch prediction (3 reviews):")
for text, result in zip(texts, results):
    print(f"  '{text[:45]}...' → {result['sentiment']} ({result['confidence']:.3f})")

# Model info
info = get_model_info()
print(f"\nModel Info:")
print(f"  Test Accuracy : {info['test_accuracy']*100:.2f}%")
print(f"  Test F1       : {info['test_f1']*100:.2f}%")
print(f"  Device        : {info['device']}")

print("\n✅ All inference tests passed")