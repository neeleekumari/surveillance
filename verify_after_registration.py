"""
Run this AFTER registering both workers to verify the fix worked
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database_module import DatabaseManager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("VERIFICATION - CHECKING NEW EMBEDDINGS")
print("=" * 70)

db = DatabaseManager()
embeddings_data = db.get_all_face_embeddings()

if len(embeddings_data) < 2:
    print(f"\n✗ Only {len(embeddings_data)} worker(s) registered")
    print("  Please register both workers first")
    sys.exit(1)

print(f"\n✓ Found {len(embeddings_data)} workers")

# Get embeddings
emb1 = embeddings_data[0]['embedding']
emb2 = embeddings_data[1]['embedding']

# Normalize
emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

# Calculate similarity
sim = cosine_similarity([emb1_norm], [emb2_norm])[0][0]
margin = 1.0 - sim

print(f"\n{embeddings_data[0]['worker_name']} vs {embeddings_data[1]['worker_name']}:")
print(f"  Similarity: {sim:.6f} ({sim*100:.2f}%)")
print(f"  Margin: {margin:.6f} ({margin*100:.2f}%)")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

if margin > 0.85:
    print(f"\n✓✓✓ PERFECT! Margin = {margin*100:.2f}%")
    print("  Fix worked! Embeddings are very different")
    print("  System will recognize correctly")
elif margin > 0.15:
    print(f"\n✓ GOOD! Margin = {margin*100:.2f}%")
    print("  Embeddings are different enough")
    print("  System should work well")
elif margin > 0.05:
    print(f"\n⚠ MARGINAL! Margin = {margin*100:.2f}%")
    print("  Embeddings are somewhat similar")
    print("  May have occasional errors")
else:
    print(f"\n✗✗✗ FAILED! Margin = {margin*100:.2f}%")
    print("  Embeddings are TOO SIMILAR")
    print("  Fix did NOT work!")
    print("\n  This means:")
    print("  - Code still has averaging bug")
    print("  - Or workers look very similar")
    print("  - Or same person registered twice")

db.close()

print("\n" + "=" * 70)
