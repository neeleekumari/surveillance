"""
Clear Database Script
---------------------
Safely clear different parts of the database with confirmation prompts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database_module import DatabaseManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_all_data(db):
    """Clear ALL data from database (nuclear option)."""
    print("\n" + "=" * 70)
    print("⚠️  WARNING: CLEAR ALL DATA")
    print("=" * 70)
    print("\nThis will DELETE:")
    print("  - All workers")
    print("  - All face embeddings")
    print("  - All face photos")
    print("  - All attendance records")
    print("  - All activity logs")
    print("\n⚠️  THIS CANNOT BE UNDONE!")
    
    confirm = input("\nType 'DELETE ALL' to confirm: ").strip()
    if confirm != "DELETE ALL":
        print("\n✓ Cancelled - no data deleted")
        return
    
    try:
        with db.conn.cursor() as cursor:
            # Delete in correct order (respecting foreign keys)
            cursor.execute("DELETE FROM attendance_records")
            attendance_count = cursor.rowcount
            
            cursor.execute("DELETE FROM activity_log")
            activity_count = cursor.rowcount
            
            cursor.execute("DELETE FROM face_photos")
            photos_count = cursor.rowcount
            
            cursor.execute("DELETE FROM face_embeddings")
            embeddings_count = cursor.rowcount
            
            cursor.execute("DELETE FROM workers")
            workers_count = cursor.rowcount
        
        print("\n" + "=" * 70)
        print("✓ DATABASE CLEARED")
        print("=" * 70)
        print(f"\nDeleted:")
        print(f"  - {workers_count} workers")
        print(f"  - {embeddings_count} face embeddings")
        print(f"  - {photos_count} face photos")
        print(f"  - {attendance_count} attendance records")
        print(f"  - {activity_count} activity logs")
        print("\n✓ Database is now empty and ready for fresh start")
        
    except Exception as e:
        print(f"\n✗ Error clearing database: {e}")

def clear_embeddings_only(db):
    """Clear only face embeddings and photos (keep worker records)."""
    print("\n" + "=" * 70)
    print("CLEAR EMBEDDINGS & PHOTOS")
    print("=" * 70)
    print("\nThis will DELETE:")
    print("  - All face embeddings")
    print("  - All face photos")
    print("\nThis will KEEP:")
    print("  - Worker records (names, IDs, positions)")
    print("  - Attendance records")
    print("  - Activity logs")
    
    confirm = input("\nType 'DELETE EMBEDDINGS' to confirm: ").strip()
    if confirm != "DELETE EMBEDDINGS":
        print("\n✓ Cancelled - no data deleted")
        return
    
    try:
        with db.conn.cursor() as cursor:
            cursor.execute("DELETE FROM face_photos")
            photos_count = cursor.rowcount
            
            cursor.execute("DELETE FROM face_embeddings")
            embeddings_count = cursor.rowcount
        
        print("\n✓ Deleted:")
        print(f"  - {embeddings_count} face embeddings")
        print(f"  - {photos_count} face photos")
        print("\n✓ Worker records preserved - you can re-register them")
        
    except Exception as e:
        print(f"\n✗ Error clearing embeddings: {e}")

def clear_workers_only(db):
    """Clear only workers without embeddings."""
    print("\n" + "=" * 70)
    print("CLEAR WORKERS WITHOUT EMBEDDINGS")
    print("=" * 70)
    
    try:
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
            return
        
        print(f"\nFound {len(workers_without_embeddings)} worker(s) without embeddings:")
        for worker in workers_without_embeddings:
            print(f"  - ID {worker[0]}: {worker[1]}")
        
        confirm = input("\nType 'DELETE' to confirm: ").strip()
        if confirm != "DELETE":
            print("\n✓ Cancelled - no data deleted")
            return
        
        with db.conn.cursor() as cursor:
            for worker in workers_without_embeddings:
                cursor.execute("DELETE FROM workers WHERE worker_id = %s", (worker[0],))
        
        print(f"\n✓ Deleted {len(workers_without_embeddings)} worker(s) without embeddings")
        
    except Exception as e:
        print(f"\n✗ Error clearing workers: {e}")

def clear_attendance_records(db):
    """Clear attendance records only."""
    print("\n" + "=" * 70)
    print("CLEAR ATTENDANCE RECORDS")
    print("=" * 70)
    
    confirm = input("\nType 'DELETE ATTENDANCE' to confirm: ").strip()
    if confirm != "DELETE ATTENDANCE":
        print("\n✓ Cancelled - no data deleted")
        return
    
    try:
        with db.conn.cursor() as cursor:
            cursor.execute("DELETE FROM attendance_records")
            count = cursor.rowcount
        
        print(f"\n✓ Deleted {count} attendance record(s)")
        
    except Exception as e:
        print(f"\n✗ Error clearing attendance: {e}")

def clear_activity_logs(db):
    """Clear activity logs only."""
    print("\n" + "=" * 70)
    print("CLEAR ACTIVITY LOGS")
    print("=" * 70)
    
    confirm = input("\nType 'DELETE LOGS' to confirm: ").strip()
    if confirm != "DELETE LOGS":
        print("\n✓ Cancelled - no data deleted")
        return
    
    try:
        with db.conn.cursor() as cursor:
            cursor.execute("DELETE FROM activity_log")
            count = cursor.rowcount
        
        print(f"\n✓ Deleted {count} activity log(s)")
        
    except Exception as e:
        print(f"\n✗ Error clearing logs: {e}")

def reset_absent_time(db):
    """Reset absent time to 0 for all workers."""
    print("\n" + "=" * 70)
    print("RESET ABSENT TIME")
    print("=" * 70)
    
    confirm = input("\nReset absent time to 0 for all workers? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n✓ Cancelled")
        return
    
    try:
        with db.conn.cursor() as cursor:
            cursor.execute("UPDATE workers SET absent_time = 0")
            count = cursor.rowcount
        
        print(f"\n✓ Reset absent time for {count} worker(s)")
        
    except Exception as e:
        print(f"\n✗ Error resetting absent time: {e}")

def delete_test_worker(db):
    """Delete the test worker (ID 999)."""
    print("\n" + "=" * 70)
    print("DELETE TEST WORKER")
    print("=" * 70)
    
    try:
        # Check if test worker exists
        with db.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM workers WHERE worker_id = 999")
            result = cursor.fetchone()
            
            if not result:
                print("\n✓ Test worker (ID 999) not found - nothing to delete")
                return
            
            print(f"\nFound test worker: {result[0]} (ID: 999)")
            confirm = input("\nDelete test worker? (y/n): ").strip().lower()
            
            if confirm != 'y':
                print("\n✓ Cancelled")
                return
            
            # Delete test worker (cascades to embeddings and photos)
            cursor.execute("DELETE FROM workers WHERE worker_id = 999")
            
        print("\n✓ Deleted test worker (ID 999)")
        
    except Exception as e:
        print(f"\n✗ Error deleting test worker: {e}")

def show_database_status(db):
    """Show current database status."""
    print("\n" + "=" * 70)
    print("DATABASE STATUS")
    print("=" * 70)
    
    try:
        # Count workers
        all_workers = db.get_all_workers()
        print(f"\nWorkers: {len(all_workers)}")
        for worker in all_workers:
            print(f"  - ID {worker[0]}: {worker[1]} ({worker[2] or 'No position'})")
        
        # Count embeddings
        embeddings_data = db.get_all_face_embeddings()
        print(f"\nFace Embeddings: {len(embeddings_data)}")
        for emb in embeddings_data:
            print(f"  - ID {emb['worker_id']}: {emb['worker_name']}")
        
        # Count photos
        with db.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM face_photos")
            photo_count = cursor.fetchone()[0]
            print(f"\nFace Photos: {photo_count}")
            
            cursor.execute("SELECT COUNT(*) FROM attendance_records")
            attendance_count = cursor.fetchone()[0]
            print(f"\nAttendance Records: {attendance_count}")
            
            cursor.execute("SELECT COUNT(*) FROM activity_log")
            activity_count = cursor.fetchone()[0]
            print(f"\nActivity Logs: {activity_count}")
        
    except Exception as e:
        print(f"\n✗ Error getting status: {e}")

def main():
    print("=" * 70)
    print("DATABASE CLEAR UTILITY")
    print("=" * 70)
    
    # Initialize database
    try:
        db = DatabaseManager()
        print("\n✓ Connected to database")
    except Exception as e:
        print(f"\n✗ Failed to connect to database: {e}")
        sys.exit(1)
    
    while True:
        print("\n" + "=" * 70)
        print("CLEAR OPTIONS")
        print("=" * 70)
        print("\n1. Clear ALL data (workers, embeddings, photos, logs)")
        print("2. Clear embeddings & photos only (keep worker records)")
        print("3. Clear workers without embeddings")
        print("4. Clear attendance records only")
        print("5. Clear activity logs only")
        print("6. Reset absent time to 0")
        print("7. Delete test worker (ID 999)")
        print("8. Show database status")
        print("9. Exit")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            clear_all_data(db)
        elif choice == "2":
            clear_embeddings_only(db)
        elif choice == "3":
            clear_workers_only(db)
        elif choice == "4":
            clear_attendance_records(db)
        elif choice == "5":
            clear_activity_logs(db)
        elif choice == "6":
            reset_absent_time(db)
        elif choice == "7":
            delete_test_worker(db)
        elif choice == "8":
            show_database_status(db)
        elif choice == "9":
            print("\n✓ Exiting...")
            break
        else:
            print("\n✗ Invalid choice")
    
    db.close()
    print("\n" + "=" * 70)
    print("✓ DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
