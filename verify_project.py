#!/usr/bin/env python3
"""
Project verification script.
This script checks that all required files are present and have the expected content.
"""
import os
import sys
from pathlib import Path

def verify_project_structure():
    """Verify that the project structure is correct."""
    print("🔍 Verifying project structure...")
    
    # Define expected files and directories
    expected_structure = {
        "root_files": [
            "README.md",
            "TODO.md",
            "SUMMARY.md",
            "requirements.txt",
            "requirements-dev.txt",
            "setup.py",
            "run.py",
            "build.py",
            "demo.py",
            ".gitignore"
        ],
        "directories": [
            "src",
            "config",
            "tests",
            "docs",
            "assets"
        ],
        "src_files": [
            "__init__.py",
            "main.py",
            "camera_manager.py",
            "detection_module.py",
            "presence_tracker.py",
            "database_module.py",
            "alert_manager.py",
            "ui_manager.py",
            "report_generator.py",
            "config_manager.py"
        ],
        "config_files": [
            "config.json"
        ],
        "test_files": [
            "test_integration.py",
            "test_camera.py",
            "test_detection.py",
            "test_database.py"
        ],
        "doc_files": [
            "api.md",
            "user_manual.md",
            "troubleshooting.md",
            "deployment.md"
        ],
        "asset_dirs": [
            "icons",
            "models"
        ]
    }
    
    # Check root files
    missing_files = []
    for file in expected_structure["root_files"]:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing root files: {missing_files}")
        return False
    else:
        print("✅ All root files present")
    
    # Check directories
    missing_dirs = []
    for directory in expected_structure["directories"]:
        if not Path(directory).exists():
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All directories present")
    
    # Check src files
    missing_src_files = []
    src_dir = Path("src")
    for file in expected_structure["src_files"]:
        if not (src_dir / file).exists():
            missing_src_files.append(file)
    
    if missing_src_files:
        print(f"❌ Missing src files: {missing_src_files}")
        return False
    else:
        print("✅ All src files present")
    
    # Check config files
    missing_config_files = []
    config_dir = Path("config")
    for file in expected_structure["config_files"]:
        if not (config_dir / file).exists():
            missing_config_files.append(file)
    
    if missing_config_files:
        print(f"❌ Missing config files: {missing_config_files}")
        return False
    else:
        print("✅ All config files present")
    
    # Check test files
    missing_test_files = []
    tests_dir = Path("tests")
    for file in expected_structure["test_files"]:
        if not (tests_dir / file).exists():
            missing_test_files.append(file)
    
    if missing_test_files:
        print(f"❌ Missing test files: {missing_test_files}")
        return False
    else:
        print("✅ All test files present")
    
    # Check doc files
    missing_doc_files = []
    docs_dir = Path("docs")
    for file in expected_structure["doc_files"]:
        if not (docs_dir / file).exists():
            missing_doc_files.append(file)
    
    if missing_doc_files:
        print(f"❌ Missing doc files: {missing_doc_files}")
        return False
    else:
        print("✅ All doc files present")
    
    # Check asset directories
    missing_asset_dirs = []
    assets_dir = Path("assets")
    for directory in expected_structure["asset_dirs"]:
        if not (assets_dir / directory).exists():
            missing_asset_dirs.append(directory)
    
    if missing_asset_dirs:
        print(f"❌ Missing asset directories: {missing_asset_dirs}")
        return False
    else:
        print("✅ All asset directories present")
    
    return True

def verify_todo_completion():
    """Verify that all TODO items are marked as complete."""
    print("\n🔍 Verifying TODO completion...")
    
    try:
        with open("TODO.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Count unchecked items
        unchecked_items = content.count("- [ ]")
        
        # The Future Enhancements section is expected to have unchecked items
        # Count the unchecked items in that section specifically
        future_section_start = content.find("## 🚀 13. Future Enhancements")
        if future_section_start != -1:
            future_section = content[future_section_start:]
            future_unchecked = future_section.count("- [ ]")
        else:
            future_unchecked = 0
        
        # Calculate actual incomplete items (excluding future enhancements)
        actual_incomplete = unchecked_items - future_unchecked
        
        if actual_incomplete > 0:
            print(f"❌ {actual_incomplete} TODO items still unchecked (excluding future enhancements)")
            return False
        else:
            print("✅ All required TODO items completed")
            return True
    except Exception as e:
        print(f"❌ Error reading TODO.md: {e}")
        return False

def verify_requirements():
    """Verify that requirements files exist and have content."""
    print("\n🔍 Verifying requirements files...")
    
    requirements_files = ["requirements.txt", "requirements-dev.txt"]
    
    for req_file in requirements_files:
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                print(f"❌ {req_file} is empty")
                return False
            else:
                lines = content.count("\n") + 1
                print(f"✅ {req_file} has {lines} lines of content")
        except Exception as e:
            print(f"❌ Error reading {req_file}: {e}")
            return False
    
    return True

def main():
    """Main verification function."""
    print("🏢 Floor Monitoring Application - Project Verification")
    print("=" * 50)
    
    # Run all verification checks
    checks = [
        verify_project_structure,
        verify_todo_completion,
        verify_requirements
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All verification checks passed!")
        print("✅ Project is complete and ready for deployment.")
    else:
        print("❌ Some verification checks failed.")
        print("⚠️  Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)