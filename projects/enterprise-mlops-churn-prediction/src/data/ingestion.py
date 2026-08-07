"""
Batch data ingestion script
Reads new data files, validates quality, appends to training data

Section A: Data & Features (25%) - Data Pipeline Component
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import yaml
import json
import argparse

from src.data.quality import DataQualityChecker, _json_default

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Batch data ingestion for churn prediction
    
    Requirements (from Instructions.txt):
    1. Read new data file(s) (e.g., daily CSV)
    2. Append/merge to training data table/file
    3. Log what was ingested (N rows, date)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.quality_checker = DataQualityChecker(self.config['monitoring'])
    
    def ingest_batch(self, input_file: str, output_file: str) -> Dict:
        """
        Ingest a batch of new data
        
        Args:
            input_file: Path to new data CSV
            output_file: Path to training data CSV
        
        Returns:
            ingestion_summary: Dict with ingestion stats
        """
        logger.info("=" * 80)
        logger.info("STARTING BATCH DATA INGESTION")
        logger.info("=" * 80)
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        
        # Read new data
        try:
            new_data = pd.read_csv(input_file)
            logger.info(f"✅ Read {len(new_data)} rows from {input_file}")
        except Exception as e:
            logger.error(f"❌ Failed to read {input_file}: {e}")
            raise
        
        # Run data quality checks
        logger.info("\nRunning data quality checks on new data...")
        quality_report = self.quality_checker.run_all_checks(new_data)
        
        if not quality_report['overall_passed']:
            logger.error("❌ Data quality checks failed. Aborting ingestion.")
            return {
                'status': 'failed',
                'reason': 'data_quality_checks_failed',
                'quality_report': quality_report
            }
        
        # Load existing training data (if exists)
        output_path = Path(output_file)
        if output_path.exists():
            try:
                existing_data = pd.read_csv(output_file)
                logger.info(f"✅ Loaded {len(existing_data)} existing rows from {output_file}")
            except Exception as e:
                logger.error(f"❌ Failed to load existing data: {e}")
                raise
        else:
            existing_data = pd.DataFrame()
            logger.info("ℹ️  No existing data found. Creating new training file.")
        
        # Merge new and existing data
        if len(existing_data) > 0:
            merged_data = pd.concat([existing_data, new_data], ignore_index=True)
            logger.info(f"✅ Merged data: {len(existing_data)} existing + {len(new_data)} new = {len(merged_data)} total")
        else:
            merged_data = new_data.copy()
            logger.info(f"✅ Using new data as initial training set: {len(merged_data)} rows")
        
        # Remove duplicates (based on customerID)
        duplicates_removed = 0
        if 'customerID' in merged_data.columns:
            before_dedup = len(merged_data)
            merged_data = merged_data.drop_duplicates(subset=['customerID'], keep='last')
            after_dedup = len(merged_data)
            duplicates_removed = before_dedup - after_dedup
            
            if duplicates_removed > 0:
                logger.info(f"ℹ️  Removed {duplicates_removed} duplicate customer IDs (kept most recent)")
        
        # Save merged data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_file, index=False)
        logger.info(f"✅ Saved {len(merged_data)} rows to {output_file}")
        
        # Create ingestion summary
        summary = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'input_file': input_file,
            'output_file': output_file,
            'new_rows': len(new_data),
            'existing_rows': len(existing_data),
            'total_rows': len(merged_data),
            'duplicates_removed': duplicates_removed,
            'quality_report': quality_report
        }
        
        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Status: {summary['status']}")
        logger.info(f"  New rows ingested: {summary['new_rows']}")
        logger.info(f"  Existing rows: {summary['existing_rows']}")
        logger.info(f"  Total rows: {summary['total_rows']}")
        logger.info(f"  Duplicates removed: {summary['duplicates_removed']}")
        logger.info(f"  Timestamp: {summary['timestamp']}")
        logger.info("=" * 80)
        
        # Save ingestion log
        log_dir = Path('artifacts/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        logger.info(f"✅ Ingestion log saved to {log_file}")
        
        return summary


def main():
    """
    Main ingestion script
    """
    parser = argparse.ArgumentParser(description='Batch data ingestion for churn prediction')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output training data CSV path')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Run ingestion
    ingestion = DataIngestion(config_path=args.config)
    summary = ingestion.ingest_batch(args.input, args.output)
    
    # Exit with appropriate code
    if summary['status'] == 'success':
        logger.info("✅ Ingestion completed successfully")
        exit(0)
    else:
        logger.error("❌ Ingestion failed")
        exit(1)


if __name__ == "__main__":
    main()
