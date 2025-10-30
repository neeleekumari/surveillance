#!/usr/bin/env python3
"""
Build script for the Floor Monitoring Application.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    try:
        import PyInstaller  # type: ignore[import-not-found]
        print("PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable...")
    
    # Create dist directory if it doesn't exist
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", "FloorMonitor",
        "--distpath", "dist",
        "--workpath", "build",
        "--specpath", "build",
        "run.py"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("Executable built successfully!")
        print(f"Executable location: {dist_dir / 'FloorMonitor.exe'}")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")
        sys.exit(1)

def create_installer():
    """Create an installer package (placeholder)."""
    print("Creating installer package...")
    print("Note: This is a placeholder. You would need to use NSIS or Inno Setup to create a real installer.")
    
    # Create a simple installer directory structure
    installer_dir = Path("installer")
    installer_dir.mkdir(exist_ok=True)
    
    # Copy executable
    dist_exe = Path("dist/FloorMonitor.exe")
    if dist_exe.exists():
        shutil.copy(dist_exe, installer_dir)
    
    # Copy config directory
    config_src = Path("config")
    if config_src.exists():
        shutil.copytree(config_src, installer_dir / "config", dirs_exist_ok=True)
    
    # Copy README and TODO
    for file in ["README.md", "TODO.md"]:
        file_path = Path(file)
        if file_path.exists():
            shutil.copy(file_path, installer_dir)
    
    print(f"Installer package created in {installer_dir}")

def main():
    """Main build function."""
    print("Floor Monitoring Application Build Script")
    print("=" * 40)
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Build executable
    build_executable()
    
    # Create installer package
    create_installer()
    
    print("\nBuild completed successfully!")

if __name__ == "__main__":
    main()