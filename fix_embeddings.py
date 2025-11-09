"""
Fix corrupted embeddings by deleting them and prompting for re-registration.
Run this script to clean up embeddings that are too similar (>40% similarity).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database_module import DatabaseManager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("FIXING CORRUPTED EMBEDDINGS")
print("=" * 70)

db = DatabaseManager()

# Get all embeddings
embeddings_data = db.get_all_face_embeddings()

if len(embeddings_data) == 0:
    print("\n✓ No embeddings found - database is clean")
    db.close()
    sys.exit(0)

print(f"\n✓ Found {len(embeddings_data)} workers")

# Check for problematic embeddings
problems_found = False
workers_to_delete = set()

if len(embeddings_data) >= 2:
    print("\nChecking embedding similarities...")
    print("-" * 70)
    
    for i in range(len(embeddings_data)):
        for j in range(i + 1, len(embeddings_data)):
            worker1 = embeddings_data[i]
            worker2 = embeddings_data[j]
            
            emb1 = worker1['embedding']
            emb2 = worker2['embedding']
            
            # Normalize
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Calculate similarity
            sim = cosine_similarity([emb1_norm], [emb2_norm])[0][0]
            margin = 1.0 - sim
            
            print(f"\n{worker1['worker_name']} vs {worker2['worker_name']}:")
            print(f"  Similarity: {sim:.6f} ({sim*100:.2f}%)")
            print(f"  Margin: {margin:.6f} ({margin*100:.2f}%)")
            
            # Check if too similar (threshold: 40% similarity = 60% margin)
            if sim > 0.40:  # More than 40% similar
                problems_found = True
                if margin < 0.15:  # Less than 15% margin - critical
                    print(f"  ⚠️  CRITICAL: Embeddings are TOO SIMILAR!")
                    print(f"      This will cause ID shuffling and false positives")
                    workers_to_delete.add(worker1['worker_id'])
                    workers_to_delete.add(worker2['worker_id'])
                elif margin < 0.40:  # Less than 40% margin - warning
                    print(f"  ⚠️  WARNING: Embeddings are quite similar")
                    print(f"      May cause occasional recognition errors")
                    workers_to_delete.add(worker1['worker_id'])
                    workers_to_delete.add(worker2['worker_id'])
                else:
                    print(f"  ✓ OK: Acceptable similarity")
            else:
                print(f"  ✓ GOOD: Embeddings are different enough")

print("\n" + "=" * 70)
print("ANALYSIS RESULTS")
print("=" * 70)

if not problems_found:
    print("\n✓✓✓ ALL GOOD!")
    print("  All embeddings are sufficiently different")
    print("  No action needed")
    db.close()
    sys.exit(0)

print(f"\n✗ PROBLEMS FOUND!")
print(f"  {len(workers_to_delete)} worker(s) have problematic embeddings")
print(f"  Workers to re-register: {sorted(workers_to_delete)}")

print("\n" + "=" * 70)
print("RECOMMENDED ACTION")
print("=" * 70)

print("\nOption 1: DELETE ALL EMBEDDINGS (Recommended)")
print("  - Deletes all face embeddings and photos")
print("  - Keeps worker records (names, IDs, positions)")
print("  - You can re-register all workers with improved validation")
print("  - This ensures all embeddings use the new quality checks")

print("\nOption 2: DELETE ONLY PROBLEMATIC WORKERS")
print(f"  - Deletes embeddings for workers: {sorted(workers_to_delete)}")
print("  - Keeps other workers unchanged")
print("  - Re-register only the problematic workers")

print("\nOption 3: DO NOTHING")
print("  - Keep current embeddings")
print("  - System will continue with poor accuracy")
print("  - Not recommended")

print("\n" + "=" * 70)

# Ask user what to do
print("\nWhat would you like to do?")
print("  1 - Delete ALL embeddings (recommended for fresh start)")
print("  2 - Delete ONLY problematic workers")
print("  3 - Do nothing (exit)")
print()

choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    print("\n" + "=" * 70)
    print("DELETING ALL EMBEDDINGS")
    print("=" * 70)
    
    try:
        with db.conn.cursor() as cursor:
            # Delete all embeddings
            cursor.execute("DELETE FROM face_embeddings")
            emb_count = cursor.rowcount
            
            # Delete all photos
            cursor.execute("DELETE FROM face_photos")
            photo_count = cursor.rowcount
            
        print(f"\n✓ Deleted {emb_count} embeddings")
        print(f"✓ Deleted {photo_count} photos")
        print(f"✓ Kept worker records (names, IDs, positions)")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Start the application:")
        print("   python run.py")
        print("\n2. Re-register ALL workers:")
        print("   - Click 'Register Worker'")
        print("   - Capture 10 high-quality photos")
        print("   - System will validate quality automatically")
        print("   - System will check uniqueness automatically")
        print("\n3. Verify after registration:")
        print("   python verify_after_registration.py")
        print("\n4. Expected result:")
        print("   ✓ GOOD! Margin = 60-90%")
        
    except Exception as e:
        print(f"\n✗ Error deleting embeddings: {e}")
        db.close()
        sys.exit(1)

elif choice == "2":
    print("\n" + "=" * 70)
    print("DELETING PROBLEMATIC WORKERS")
    print("=" * 70)
    
    try:
        for worker_id in sorted(workers_to_delete):
            # Get worker name
            worker_name = next(
                (w['worker_name'] for w in embeddings_data if w['worker_id'] == worker_id),
                f"Worker {worker_id}"
            )
            
            with db.conn.cursor() as cursor:
                # Delete embeddings
                cursor.execute(
                    "DELETE FROM face_embeddings WHERE worker_id = %s",
                    (worker_id,)
                )
                emb_count = cursor.rowcount
                
                # Delete photos
                cursor.execute(
                    "DELETE FROM face_photos WHERE worker_id = %s",
                    (worker_id,)
                )
                photo_count = cursor.rowcount
                
            print(f"\n✓ Deleted {worker_name} (ID: {worker_id})")
            print(f"  - {emb_count} embeddings deleted")
            print(f"  - {photo_count} photos deleted")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print(f"\n1. Re-register these workers:")
        for worker_id in sorted(workers_to_delete):
            worker_name = next(
                (w['worker_name'] for w in embeddings_data if w['worker_id'] == worker_id),
                f"Worker {worker_id}"
            )
            print(f"   - {worker_name} (ID: {worker_id})")
        print("\n2. Use improved registration:")
        print("   - Capture 10 high-quality photos")
        print("   - Different angles and expressions")
        print("   - Good lighting and focus")
        print("\n3. Verify after registration:")
        print("   python verify_after_registration.py")
        
    except Exception as e:
        print(f"\n✗ Error deleting workers: {e}")
        db.close()
        sys.exit(1)

elif choice == "3":
    print("\n✓ No changes made")
    print("  Embeddings remain unchanged")
    print("  System will continue with current accuracy")
    db.close()
    sys.exit(0)

else:
    print("\n✗ Invalid choice")
    db.close()
    sys.exit(1)

db.close()

print("\n" + "=" * 70)
print("✓ DONE!")
print("=" * 70)
