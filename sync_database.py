"""
Database Synchronization Script
--------------------------------
Syncs worker status between face recognition embeddings, presence tracker, and database.
Ensures database reflects current worker registration status and cleans up inconsistencies.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database_module import DatabaseManager
from src.face_recognition_module import FaceRecognitionSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 70)
print("DATABASE SYNCHRONIZATION")
print("=" * 70)

# Initialize systems
db = DatabaseManager()
face_system = FaceRecognitionSystem()  # Creates its own db_manager internally

print("\n" + "=" * 70)
print("STEP 1: CHECKING WORKERS IN DATABASE")
print("=" * 70)

# Get all workers from database
all_workers = db.get_all_workers()
print(f"\n✓ Found {len(all_workers)} workers in database:")
for worker in all_workers:
    worker_id, name, position, contact, is_active, absent_time = worker
    status = "ACTIVE" if is_active else "INACTIVE"
    print(f"  - ID {worker_id}: {name} ({position}) - {status} - Absent: {absent_time} min")

print("\n" + "=" * 70)
print("STEP 2: CHECKING FACE EMBEDDINGS")
print("=" * 70)

# Get all embeddings
embeddings_data = db.get_all_face_embeddings()
print(f"\n✓ Found {len(embeddings_data)} workers with face embeddings:")

workers_with_embeddings = set()
for emb in embeddings_data:
    worker_id = emb['worker_id']
    worker_name = emb['worker_name']
    workers_with_embeddings.add(worker_id)
    print(f"  - ID {worker_id}: {worker_name} - Has embedding")

print("\n" + "=" * 70)
print("STEP 3: CHECKING FACE PHOTOS")
print("=" * 70)

workers_with_photos = {}
for worker in all_workers:
    worker_id = worker[0]
    photos = db.get_face_photos(worker_id)
    if photos:
        workers_with_photos[worker_id] = len(photos)
        print(f"  - ID {worker_id}: {worker[1]} - {len(photos)} photos")

if not workers_with_photos:
    print("\n  No face photos found in database")

print("\n" + "=" * 70)
print("STEP 4: IDENTIFYING INCONSISTENCIES")
print("=" * 70)

inconsistencies = []

# Check 1: Workers without embeddings
workers_without_embeddings = []
for worker in all_workers:
    worker_id = worker[0]
    if worker_id not in workers_with_embeddings:
        workers_without_embeddings.append(worker)
        inconsistencies.append(f"Worker ID {worker_id} ({worker[1]}) has no face embeddings")

if workers_without_embeddings:
    print(f"\n⚠️  {len(workers_without_embeddings)} worker(s) without embeddings:")
    for worker in workers_without_embeddings:
        print(f"  - ID {worker[0]}: {worker[1]} ({worker[2]})")
else:
    print("\n✓ All active workers have embeddings")

# Check 2: Embeddings without worker records (orphaned embeddings)
worker_ids_in_db = {w[0] for w in all_workers}
orphaned_embeddings = []
for emb in embeddings_data:
    if emb['worker_id'] not in worker_ids_in_db:
        orphaned_embeddings.append(emb)
        inconsistencies.append(f"Embedding for worker ID {emb['worker_id']} has no worker record")

if orphaned_embeddings:
    print(f"\n⚠️  {len(orphaned_embeddings)} orphaned embedding(s):")
    for emb in orphaned_embeddings:
        print(f"  - ID {emb['worker_id']}: {emb['worker_name']} (no worker record)")
else:
    print("\n✓ No orphaned embeddings found")

# Check 3: Workers with embeddings but no photos
workers_with_emb_no_photos = []
for worker_id in workers_with_embeddings:
    if worker_id not in workers_with_photos:
        worker_name = next((w[1] for w in all_workers if w[0] == worker_id), "Unknown")
        workers_with_emb_no_photos.append((worker_id, worker_name))
        inconsistencies.append(f"Worker ID {worker_id} ({worker_name}) has embeddings but no photos")

if workers_with_emb_no_photos:
    print(f"\n⚠️  {len(workers_with_emb_no_photos)} worker(s) with embeddings but no photos:")
    for worker_id, worker_name in workers_with_emb_no_photos:
        print(f"  - ID {worker_id}: {worker_name}")
else:
    print("\n✓ All workers with embeddings have photos")

# Check 4: Inactive workers with embeddings
inactive_with_embeddings = []
for worker in all_workers:
    worker_id, name, position, contact, is_active, absent_time = worker
    if not is_active and worker_id in workers_with_embeddings:
        inactive_with_embeddings.append(worker)
        inconsistencies.append(f"Inactive worker ID {worker_id} ({name}) still has embeddings")

if inactive_with_embeddings:
    print(f"\n⚠️  {len(inactive_with_embeddings)} inactive worker(s) with embeddings:")
    for worker in inactive_with_embeddings:
        print(f"  - ID {worker[0]}: {worker[1]}")
else:
    print("\n✓ No inactive workers with embeddings")

print("\n" + "=" * 70)
print("STEP 5: SYNCHRONIZATION SUMMARY")
print("=" * 70)

if not inconsistencies:
    print("\n✓✓✓ DATABASE IS SYNCHRONIZED!")
    print("  - All workers have proper embeddings")
    print("  - No orphaned data")
    print("  - No inconsistencies found")
    db.close()
    sys.exit(0)

print(f"\n⚠️  Found {len(inconsistencies)} inconsistency(ies):")
for i, issue in enumerate(inconsistencies, 1):
    print(f"  {i}. {issue}")

print("\n" + "=" * 70)
print("STEP 6: SYNCHRONIZATION OPTIONS")
print("=" * 70)

print("\nWhat would you like to do?")
print("\n1. CLEAN ORPHANED DATA")
print("   - Delete embeddings without worker records")
print("   - Delete photos without worker records")
print("   - Keeps all valid worker data")

print("\n2. DELETE WORKERS WITHOUT EMBEDDINGS")
print("   - Remove worker records that have no face embeddings")
print("   - Useful if workers were added but never registered")
print("   - WARNING: This deletes worker records permanently")

print("\n3. DEACTIVATE WORKERS WITHOUT EMBEDDINGS")
print("   - Mark workers without embeddings as inactive")
print("   - Keeps worker records but marks them as not active")
print("   - Safer than deletion")

print("\n4. RESET ABSENT TIME FOR ALL WORKERS")
print("   - Set absent_time to 0 for all workers")
print("   - Useful for fresh start of absence tracking")

print("\n5. SHOW DETAILED WORKER STATUS")
print("   - Display complete status for each worker")
print("   - Includes embeddings, photos, activity")

print("\n6. DO NOTHING")
print("   - Exit without making changes")

print("\n" + "=" * 70)

choice = input("\nEnter choice (1-6): ").strip()

if choice == "1":
    print("\n" + "=" * 70)
    print("CLEANING ORPHANED DATA")
    print("=" * 70)
    
    try:
        deleted_embeddings = 0
        deleted_photos = 0
        
        # Delete orphaned embeddings
        for emb in orphaned_embeddings:
            with db.conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM face_embeddings WHERE worker_id = %s",
                    (emb['worker_id'],)
                )
                deleted_embeddings += cursor.rowcount
                
                cursor.execute(
                    "DELETE FROM face_photos WHERE worker_id = %s",
                    (emb['worker_id'],)
                )
                deleted_photos += cursor.rowcount
        
        print(f"\n✓ Deleted {deleted_embeddings} orphaned embedding(s)")
        print(f"✓ Deleted {deleted_photos} orphaned photo(s)")
        print("✓ Database cleaned successfully")
        
    except Exception as e:
        print(f"\n✗ Error cleaning data: {e}")

elif choice == "2":
    print("\n" + "=" * 70)
    print("DELETING WORKERS WITHOUT EMBEDDINGS")
    print("=" * 70)
    
    if not workers_without_embeddings:
        print("\n✓ No workers to delete")
    else:
        print(f"\n⚠️  WARNING: This will permanently delete {len(workers_without_embeddings)} worker(s):")
        for worker in workers_without_embeddings:
            print(f"  - ID {worker[0]}: {worker[1]}")
        
        confirm = input("\nType 'DELETE' to confirm: ").strip()
        if confirm == "DELETE":
            try:
                for worker in workers_without_embeddings:
                    with db.conn.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM workers WHERE worker_id = %s",
                            (worker[0],)
                        )
                print(f"\n✓ Deleted {len(workers_without_embeddings)} worker(s)")
            except Exception as e:
                print(f"\n✗ Error deleting workers: {e}")
        else:
            print("\n✓ Deletion cancelled")

elif choice == "3":
    print("\n" + "=" * 70)
    print("DEACTIVATING WORKERS WITHOUT EMBEDDINGS")
    print("=" * 70)
    
    if not workers_without_embeddings:
        print("\n✓ No workers to deactivate")
    else:
        try:
            for worker in workers_without_embeddings:
                with db.conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE workers SET is_active = false WHERE worker_id = %s",
                        (worker[0],)
                    )
            print(f"\n✓ Deactivated {len(workers_without_embeddings)} worker(s)")
            print("  Workers are now marked as inactive but records are preserved")
        except Exception as e:
            print(f"\n✗ Error deactivating workers: {e}")

elif choice == "4":
    print("\n" + "=" * 70)
    print("RESETTING ABSENT TIME")
    print("=" * 70)
    
    try:
        with db.conn.cursor() as cursor:
            cursor.execute("UPDATE workers SET absent_time = 0")
            count = cursor.rowcount
        print(f"\n✓ Reset absent_time to 0 for {count} worker(s)")
        print("  Fresh start for absence tracking")
    except Exception as e:
        print(f"\n✗ Error resetting absent time: {e}")

elif choice == "5":
    print("\n" + "=" * 70)
    print("DETAILED WORKER STATUS")
    print("=" * 70)
    
    for worker in all_workers:
        worker_id, name, position, contact, is_active, absent_time = worker
        
        print(f"\n{'='*70}")
        print(f"Worker ID: {worker_id}")
        print(f"Name: {name}")
        print(f"Position: {position}")
        print(f"Contact: {contact}")
        print(f"Status: {'ACTIVE' if is_active else 'INACTIVE'}")
        print(f"Absent Time: {absent_time} minutes")
        
        # Check embeddings
        has_embedding = worker_id in workers_with_embeddings
        print(f"Has Embedding: {'YES' if has_embedding else 'NO'}")
        
        # Check photos
        photo_count = workers_with_photos.get(worker_id, 0)
        print(f"Photos: {photo_count}")
        
        # Get recent activity
        try:
            activities = db.get_worker_activities(worker_id, limit=5)
            if activities:
                print(f"Recent Activity:")
                for act in activities[:3]:
                    print(f"  - {act['status']} at {act['timestamp']}")
            else:
                print("Recent Activity: None")
        except Exception:
            print("Recent Activity: Error fetching")
        
        # Overall status
        if has_embedding and photo_count > 0 and is_active:
            print("Overall: ✓ READY FOR RECOGNITION")
        elif not has_embedding:
            print("Overall: ⚠️  NEEDS REGISTRATION (no embedding)")
        elif not is_active:
            print("Overall: ⚠️  INACTIVE")
        else:
            print("Overall: ⚠️  INCOMPLETE SETUP")

elif choice == "6":
    print("\n✓ No changes made")
else:
    print("\n✗ Invalid choice")

db.close()

print("\n" + "=" * 70)
print("✓ SYNCHRONIZATION COMPLETE")
print("=" * 70)
