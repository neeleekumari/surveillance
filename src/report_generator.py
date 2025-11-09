"""
Report Generator Module
---------------------
Generates reports and visualizations for the floor monitoring system.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates reports and visualizations for worker presence data."""
    
    def __init__(self, db_manager: Optional[Any] = None, config: Optional[dict] = None):
        """Initialize the report generator.
        
        Args:
            db_manager: DatabaseManager instance (optional)
            config: Configuration dictionary
        """
        self.db_manager = db_manager
        self.config = config or {}
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("ReportGenerator initialized")
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a daily report for worker presence.
        
        Args:
            date: Date for the report (defaults to today)
            
        Returns:
            Dictionary containing report data
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Get all workers
            # Note: This would require a method to get all workers from the database
            # For now, we'll simulate this data
            
            report_data = {
                "date": date.strftime("%Y-%m-%d"),
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_workers": 0,
                    "present_workers": 0,
                    "absent_workers": 0,
                    "average_presence_time": 0
                },
                "workers": []
            }
            
            logger.info(f"Generated daily report for {date.strftime('%Y-%m-%d')}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise
    
    def generate_weekly_report(self, start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a weekly report for worker presence.
        
        Args:
            start_date: Start date for the week (defaults to previous Monday)
            
        Returns:
            Dictionary containing report data
        """
        if start_date is None:
            # Calculate previous Monday
            today = datetime.now()
            start_date = today - timedelta(days=today.weekday())
        
        end_date = start_date + timedelta(days=6)
        
        try:
            report_data = {
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_workers": 0,
                    "average_daily_present": 0,
                    "total_presence_hours": 0
                },
                "daily_breakdown": [],
                "workers": []
            }
            
            logger.info(f"Generated weekly report for {report_data['period']}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {str(e)}")
            raise
    
    def export_to_csv(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export report data to CSV format.
        
        Args:
            report_data: Report data dictionary
            filename: Output filename (defaults to auto-generated)
            
        Returns:
            Path to the generated CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.csv"
        
        filepath = self.reports_dir / filename
        
        try:
            # Extract worker data
            workers = report_data.get("workers", [])
            
            if not workers:
                logger.warning("No worker data to export to CSV")
                return str(filepath)
            
            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = workers[0].keys() if workers else []
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for worker in workers:
                    writer.writerow(worker)
            
            logger.info(f"Exported report to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def export_to_pdf(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export report data to PDF format with visualizations.
        
        Args:
            report_data: Report data dictionary
            filename: Output filename (defaults to auto-generated)
            
        Returns:
            Path to the generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.pdf"
        
        filepath = self.reports_dir / filename
        
        try:
            with PdfPages(filepath) as pdf:
                # Create summary page
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.1, 0.8, "Floor Monitoring Report", fontsize=20, fontweight='bold')
                
                summary = report_data.get("summary", {})
                summary_text = "\n".join([
                    f"Report Period: {report_data.get('date', report_data.get('period', 'N/A'))}",
                    f"Generated: {report_data.get('generated_at', 'N/A')}",
                    f"Total Workers: {summary.get('total_workers', 0)}",
                    f"Present Workers: {summary.get('present_workers', 0)}",
                    f"Absent Workers: {summary.get('absent_workers', 0)}",
                    f"Average Presence Time: {summary.get('average_presence_time', 0):.2f} hours"
                ])
                
                ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
                
                # Create visualization page (if we had data)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "Worker Presence Visualization\n(Placeholder)", 
                       fontsize=16, ha='center', va='center')
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
            
            logger.info(f"Exported report to PDF: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            raise
    
    def get_worker_statistics(self, worker_id: int, days: int = 30) -> Dict[str, Any]:
        """Get statistics for a specific worker over a period.
        
        Args:
            worker_id: Worker ID
            days: Number of days to analyze
            
        Returns:
            Dictionary containing worker statistics
        """
        try:
            stats = {
                "worker_id": worker_id,
                "total_days": days,
                "days_present": 0,
                "total_hours": 0,
                "average_daily_hours": 0,
                "longest_session": 0,
                "most_recent_activity": None
            }
            
            # Get worker activities from database if available
            if self.db_manager:
                activities = self.db_manager.get_worker_activities(worker_id, limit=1000)
                if activities:
                    stats["most_recent_activity"] = activities[0]  # Most recent is first
            else:
                logger.warning("No database manager available for worker statistics")
            
            logger.info(f"Generated statistics for worker {worker_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting worker statistics: {str(e)}")
            raise


def test_report_generator():
    """Test function for the ReportGenerator class."""
    import sys
    from pathlib import Path
    import time
    sys.path.append(str(Path(__file__).parent))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a mock database manager for testing
    class MockDBManager:
        def get_worker_activities(self, worker_id, limit=100):
            return [
                {
                    "log_id": 1,
                    "status": "present",
                    "timestamp": datetime.now(),
                    "duration_seconds": 3600
                }
            ]
    
    # Create report generator
    db_manager = MockDBManager()
    config = {}
    report_gen = ReportGenerator(db_manager, config)
    
    print("Testing ReportGenerator...")
    
    try:
        # Generate daily report
        daily_report = report_gen.generate_daily_report()
        print(f"Daily report generated: {daily_report['date']}")
        
        # Generate weekly report
        weekly_report = report_gen.generate_weekly_report()
        print(f"Weekly report generated: {weekly_report['period']}")
        
        # Export to CSV
        csv_path = report_gen.export_to_csv(daily_report)
        print(f"Exported to CSV: {csv_path}")
        
        # Export to PDF
        pdf_path = report_gen.export_to_pdf(daily_report)
        print(f"Exported to PDF: {pdf_path}")
        
        # Get worker statistics
        worker_stats = report_gen.get_worker_statistics(1)
        print(f"Worker statistics: {worker_stats}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("ReportGenerator test completed.")


if __name__ == "__main__":
    test_report_generator()