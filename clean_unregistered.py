"""
Quick script to delete workers without embeddings
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database_module import DatabaseManager

print("=" * 70)
print("CLEANING WORKERS WITHOUT EMBEDDINGS")
print("=" * 70)

db = DatabaseManager()

# Get workers without embeddings
embeddings_data = db.get_all_face_embeddings()
workers_with_embeddings = {emb['worker_id'] for emb in embeddings_data}

all_workers = db.get_all_workers()
workers_without_embeddings = [
    w for w in all_workers 
    if w[0] not in workers_with_embeddings
]

if not workers_without_embeddings:
    print("\n✓ No workers without embeddings found")
else:
    print(f"\nFound {len(workers_without_embeddings)} worker(s) without embeddings:")
    for worker in workers_without_embeddings:
        print(f"  - ID {worker[0]}: {worker[1]}")
    
    print("\nDeleting workers without embeddings...")
    with db.conn.cursor() as cursor:
        for worker in workers_without_embeddings:
            cursor.execute("DELETE FROM workers WHERE worker_id = %s", (worker[0],))
            print(f"  ✓ Deleted: {worker[1]} (ID: {worker[0]})")
    
    print(f"\n✓ Deleted {len(workers_without_embeddings)} worker(s)")

db.close()
print("\n" + "=" * 70)
print("✓ CLEANUP COMPLETE")
print("=" * 70)
