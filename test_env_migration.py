"""
Test script to verify environment variable migration
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_env_file_exists():
    """Test 1: Check if .env file exists"""
    print("\n" + "="*70)
    print("TEST 1: .env File Exists")
    print("="*70)
    
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print("✅ PASS: .env file exists")
        return True
    else:
        print("❌ FAIL: .env file not found")
        return False

def test_password_not_in_config():
    """Test 2: Verify password is not in config.json"""
    print("\n" + "="*70)
    print("TEST 2: Password Not in config.json")
    print("="*70)
    
    config_path = Path(__file__).parent / "config" / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            content = f.read()
            if '"password"' in content.lower():
                print("❌ FAIL: Password field found in config.json")
                print("   (Password should be removed from config.json)")
                return False
            else:
                print("✅ PASS: No password field in config.json")
                return True
    else:
        print("⚠️  WARNING: config.json not found")
        return False

def test_env_variables_loaded():
    """Test 3: Check if environment variables are loaded"""
    print("\n" + "="*70)
    print("TEST 3: Environment Variables Loaded")
    print("="*70)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        
        results = []
        
        if db_password:
            print(f"✅ DB_PASSWORD is set (length: {len(db_password)} chars)")
            results.append(True)
        else:
            print("❌ DB_PASSWORD is not set")
            results.append(False)
        
        if db_host:
            print(f"✅ DB_HOST is set: {db_host}")
            results.append(True)
        else:
            print("❌ DB_HOST is not set")
            results.append(False)
        
        if db_name:
            print(f"✅ DB_NAME is set: {db_name}")
            results.append(True)
        else:
            print("❌ DB_NAME is not set")
            results.append(False)
        
        return all(results)
        
    except ImportError:
        print("❌ FAIL: python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False

def test_config_manager():
    """Test 4: Test ConfigManager loads password from .env"""
    print("\n" + "="*70)
    print("TEST 4: ConfigManager Loads from .env")
    print("="*70)
    
    try:
        from config_manager import ConfigManager
        
        config = ConfigManager()
        db_config = config.get_database_config()
        
        print(f"Database Host: {db_config.get('host')}")
        print(f"Database Name: {db_config.get('name')}")
        print(f"Database User: {db_config.get('user')}")
        print(f"Database Port: {db_config.get('port')}")
        
        if db_config.get('password'):
            print(f"✅ Password loaded from .env (length: {len(db_config['password'])} chars)")
            print("✅ PASS: ConfigManager successfully loads password from environment")
            return True
        else:
            print("❌ FAIL: Password not loaded")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error loading ConfigManager: {e}")
        return False

def test_database_connection():
    """Test 5: Test actual database connection"""
    print("\n" + "="*70)
    print("TEST 5: Database Connection")
    print("="*70)
    
    try:
        from database_module import DatabaseManager
        
        print("Attempting to connect to database...")
        db = DatabaseManager()
        
        if db.conn:
            print("✅ PASS: Database connection successful!")
            print("✅ Password from .env works correctly")
            
            # Test a simple query
            with db.conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"PostgreSQL version: {version[0][:50]}...")
            
            return True
        else:
            print("❌ FAIL: Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Database connection error: {e}")
        print("   This might be expected if PostgreSQL is not running")
        return False

def run_all_tests():
    """Run all tests and display summary"""
    print("\n" + "="*70)
    print("ENVIRONMENT VARIABLE MIGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("1. .env File Exists", test_env_file_exists),
        ("2. Password Not in config.json", test_password_not_in_config),
        ("3. Environment Variables Loaded", test_env_variables_loaded),
        ("4. ConfigManager Loads from .env", test_config_manager),
        ("5. Database Connection", test_database_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Migration successful!")
    elif passed >= 4:
        print("\n⚠️  Most tests passed. Check failed tests above.")
    else:
        print("\n❌ Migration incomplete. Please review errors above.")
    
    print("\n" + "="*70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
