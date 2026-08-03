"""
Retraining trigger logic
Determines when to retrain the model based on multiple signals

Section D: Monitoring, Data Quality and Retraining Trigger (25%) - Retraining Logic
"""

import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Retraining trigger logic
    
    Implements 3 retraining signals (from Instructions.txt):
    1. New data volume threshold
    2. Performance degradation
    3. Feature drift detection
    
    Optional 4th signal:
    4. Time-based (periodic retraining)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.retraining_config = config['retraining']
    
    def should_retrain(self, metrics: Dict) -> Tuple[bool, str, Dict]:
        """
        Determine if model should be retrained
        
        Args:
            metrics: Dict containing:
                - new_labeled_data_count: Number of new labeled samples
                - current_auc: Current model AUC on recent data
                - baseline_auc: Baseline model AUC
                - drift_score: Maximum drift score across features
                - days_since_training: Days since last training
        
        Returns:
            (should_retrain, reason, trigger_details)
        """
        logger.info("=" * 80)
        logger.info("RETRAINING TRIGGER EVALUATION")
        logger.info("=" * 80)
        
        triggers = []
        
        # Signal 1: New data volume
        min_new_data = self.retraining_config['min_new_data_count']
        new_data_count = metrics.get('new_labeled_data_count', 0)
        
        logger.info(f"\n[Signal 1] New Data Volume:")
        logger.info(f"  New labeled data: {new_data_count}")
        logger.info(f"  Threshold: {min_new_data}")
        
        if new_data_count >= min_new_data:
            reason = f"Sufficient new labeled data: {new_data_count} >= {min_new_data}"
            logger.info(f"  ✅ TRIGGERED: {reason}")
            triggers.append({
                'signal': 'new_data_volume',
                'triggered': True,
                'reason': reason,
                'value': new_data_count,
                'threshold': min_new_data
            })
        else:
            logger.info(f"  ⏸️  Not triggered: {new_data_count} < {min_new_data}")
            triggers.append({
                'signal': 'new_data_volume',
                'triggered': False,
                'value': new_data_count,
                'threshold': min_new_data
            })
        
        # Signal 2: Performance degradation
        max_auc_drop = self.retraining_config['max_auc_drop']
        current_auc = metrics.get('current_auc', None)
        baseline_auc = metrics.get('baseline_auc', None)
        
        logger.info(f"\n[Signal 2] Performance Degradation:")
        logger.info(f"  Current AUC: {current_auc}")
        logger.info(f"  Baseline AUC: {baseline_auc}")
        logger.info(f"  Max allowed drop: {max_auc_drop}")
        
        if current_auc is not None and baseline_auc is not None:
            auc_drop = baseline_auc - current_auc
            logger.info(f"  Actual drop: {auc_drop:.4f}")
            
            if auc_drop > max_auc_drop:
                reason = f"AUC dropped by {auc_drop:.4f} (> {max_auc_drop})"
                logger.info(f"  ✅ TRIGGERED: {reason}")
                triggers.append({
                    'signal': 'performance_degradation',
                    'triggered': True,
                    'reason': reason,
                    'auc_drop': auc_drop,
                    'threshold': max_auc_drop
                })
            else:
                logger.info(f"  ⏸️  Not triggered: {auc_drop:.4f} <= {max_auc_drop}")
                triggers.append({
                    'signal': 'performance_degradation',
                    'triggered': False,
                    'auc_drop': auc_drop,
                    'threshold': max_auc_drop
                })
        else:
            logger.info(f"  ⏸️  Not triggered: Missing AUC metrics")
            triggers.append({
                'signal': 'performance_degradation',
                'triggered': False,
                'reason': 'Missing AUC metrics'
            })
        
        # Signal 3: Feature drift
        max_drift_score = self.retraining_config['max_drift_score']
        drift_score = metrics.get('drift_score', 0)
        
        logger.info(f"\n[Signal 3] Feature Drift:")
        logger.info(f"  Drift score: {drift_score:.4f}")
        logger.info(f"  Threshold: {max_drift_score}")
        
        if drift_score > max_drift_score:
            reason = f"Significant feature drift detected: {drift_score:.4f} > {max_drift_score}"
            logger.info(f"  ✅ TRIGGERED: {reason}")
            triggers.append({
                'signal': 'feature_drift',
                'triggered': True,
                'reason': reason,
                'value': drift_score,
                'threshold': max_drift_score
            })
        else:
            logger.info(f"  ⏸️  Not triggered: {drift_score:.4f} <= {max_drift_score}")
            triggers.append({
                'signal': 'feature_drift',
                'triggered': False,
                'value': drift_score,
                'threshold': max_drift_score
            })
        
        # Signal 4: Time-based (optional)
        max_days = self.retraining_config['max_days_since_training']
        days_since_training = metrics.get('days_since_training', 0)
        
        logger.info(f"\n[Signal 4] Time-Based:")
        logger.info(f"  Days since training: {days_since_training}")
        logger.info(f"  Threshold: {max_days} days")
        
        if days_since_training >= max_days:
            reason = f"Periodic retraining due: {days_since_training} >= {max_days} days"
            logger.info(f"  ✅ TRIGGERED: {reason}")
            triggers.append({
                'signal': 'time_based',
                'triggered': True,
                'reason': reason,
                'value': days_since_training,
                'threshold': max_days
            })
        else:
            logger.info(f"  ⏸️  Not triggered: {days_since_training} < {max_days} days")
            triggers.append({
                'signal': 'time_based',
                'triggered': False,
                'value': days_since_training,
                'threshold': max_days
            })
        
        # Decision
        triggered_signals = [t for t in triggers if t['triggered']]
        should_retrain = len(triggered_signals) > 0
        
        logger.info("\n" + "=" * 80)
        if should_retrain:
            logger.info(f"🔄 RETRAINING DECISION: YES")
            logger.info(f"   Triggered signals: {len(triggered_signals)}/{len(triggers)}")
            for trigger in triggered_signals:
                logger.info(f"   - {trigger['signal']}: {trigger['reason']}")
            primary_reason = triggered_signals[0]['reason']
        else:
            logger.info(f"⏸️  RETRAINING DECISION: NO")
            logger.info(f"   No signals triggered")
            primary_reason = "No retraining signals triggered"
        logger.info("=" * 80)
        
        trigger_details = {
            'timestamp': datetime.now().isoformat(),
            'should_retrain': should_retrain,
            'primary_reason': primary_reason,
            'triggered_signals': len(triggered_signals),
            'total_signals': len(triggers),
            'signals': triggers,
            'input_metrics': metrics
        }
        
        return should_retrain, primary_reason, trigger_details
    
    def save_trigger_log(self, trigger_details: Dict, output_file: str):
        """
        Save retraining trigger log
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(trigger_details, f, indent=2)
        
        logger.info(f"✅ Trigger log saved to {output_file}")


def main():
    """
    Test retraining trigger logic
    """
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trigger
    trigger = RetrainingTrigger(config)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: New data volume',
            'metrics': {
                'new_labeled_data_count': 1500,
                'current_auc': 0.82,
                'baseline_auc': 0.83,
                'drift_score': 0.15,
                'days_since_training': 10
            }
        },
        {
            'name': 'Scenario 2: Performance degradation',
            'metrics': {
                'new_labeled_data_count': 500,
                'current_auc': 0.74,
                'baseline_auc': 0.82,
                'drift_score': 0.10,
                'days_since_training': 15
            }
        },
        {
            'name': 'Scenario 3: Feature drift',
            'metrics': {
                'new_labeled_data_count': 300,
                'current_auc': 0.81,
                'baseline_auc': 0.82,
                'drift_score': 0.35,
                'days_since_training': 20
            }
        },
        {
            'name': 'Scenario 4: No triggers',
            'metrics': {
                'new_labeled_data_count': 200,
                'current_auc': 0.82,
                'baseline_auc': 0.82,
                'drift_score': 0.08,
                'days_since_training': 5
            }
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n\n{'='*80}")
        logger.info(f"TESTING: {scenario['name']}")
        logger.info(f"{'='*80}")
        
        should_retrain, reason, details = trigger.should_retrain(scenario['metrics'])
        
        # Save log
        log_file = f"artifacts/logs/retraining_trigger_{scenario['name'].replace(' ', '_').replace(':', '')}.json"
        trigger.save_trigger_log(details, log_file)
    
    logger.info("\n✅ Retraining trigger test completed!")


if __name__ == "__main__":
    main()
