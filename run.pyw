#!/usr/bin/env python3
"""
Runner script for the Floor Monitoring Application.
"""
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('floor_monitor.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Preload PyTorch to avoid DLL initialization issues
        import torch
        logger.info(f"PyTorch {torch.__version__} loaded successfully")
        
        # Import and run the main application
        from src.main import main as app_main
        logger.info("Starting Floor Monitoring Application")
        app_main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()