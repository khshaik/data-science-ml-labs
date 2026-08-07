"""
Model evaluation and comparison
Implements promotion guardrails for model selection

Section B: Model Training & Offline Evaluation (25%) - Evaluation Component
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation and comparison with promotion guardrails
    
    Promotion Rules (from Instructions.txt):
    - Only promote if AUC ≥ 0.8
    - Only promote if Recall ≥ 0.75
    - Must not reduce validation AUC relative to the simpler baseline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.promotion_rules = config['models']['promotion']
    
    def load_evaluation_reports(self) -> Tuple[Dict, Dict]:
        """
        Load evaluation reports for baseline and candidate models
        
        Returns:
            (baseline_report, candidate_report)
        """
        eval_dir = Path('artifacts/eval')
        
        baseline_path = eval_dir / 'baseline_evaluation.json'
        candidate_path = eval_dir / 'candidate_evaluation.json'
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline evaluation report not found: {baseline_path}")
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate evaluation report not found: {candidate_path}")
        
        with open(baseline_path, 'r') as f:
            baseline_report = json.load(f)
        
        with open(candidate_path, 'r') as f:
            candidate_report = json.load(f)
        
        logger.info("✅ Loaded evaluation reports")
        return baseline_report, candidate_report
    
    def compare_models(self, baseline_report: Dict, candidate_report: Dict) -> Dict:
        """
        Compare baseline and candidate models
        
        Args:
            baseline_report: Baseline model evaluation report
            candidate_report: Candidate model evaluation report
        
        Returns:
            comparison: Dict with comparison results
        """
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        
        # Select/promote on validation data. The test set remains an untouched
        # final estimate and is reported separately.
        baseline_metrics = baseline_report['validation_metrics']
        candidate_metrics = candidate_report['validation_metrics']
        
        # Create comparison table
        comparison = {
            'baseline': baseline_metrics,
            'candidate': candidate_metrics,
            'differences': {},
            'selection_dataset': 'validation',
            'final_test': {
                'baseline': baseline_report.get('test_metrics'),
                'candidate': candidate_report.get('test_metrics')
            }
        }
        
        # Calculate differences
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            diff = candidate_metrics[metric] - baseline_metrics[metric]
            comparison['differences'][metric] = diff
        
        # Log comparison
        logger.info("\nMetric Comparison (Validation Set):")
        logger.info("-" * 80)
        logger.info(f"{'Metric':<15} {'Baseline':<12} {'Candidate':<12} {'Difference':<12} {'Winner'}")
        logger.info("-" * 80)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            baseline_val = baseline_metrics[metric]
            candidate_val = candidate_metrics[metric]
            diff = comparison['differences'][metric]
            winner = "Candidate" if diff > 0 else "Baseline" if diff < 0 else "Tie"
            
            logger.info(f"{metric.upper():<15} {baseline_val:<12.4f} {candidate_val:<12.4f} "
                       f"{diff:+<12.4f} {winner}")
        
        logger.info("-" * 80)
        
        return comparison
    
    def check_promotion_guardrails(self, baseline_report: Dict, candidate_report: Dict) -> Tuple[bool, str]:
        """
        Check if candidate model meets promotion guardrails
        
        Args:
            baseline_report: Baseline model evaluation report
            candidate_report: Candidate model evaluation report
        
        Returns:
            (should_promote, reason)
        """
        logger.info("\n" + "=" * 80)
        logger.info("PROMOTION GUARDRAILS CHECK")
        logger.info("=" * 80)
        
        baseline_metrics = baseline_report['validation_metrics']
        candidate_metrics = candidate_report['validation_metrics']
        
        # Guardrail 1: Minimum AUC threshold
        min_auc = self.promotion_rules['min_auc']
        if candidate_metrics['auc'] < min_auc:
            reason = f"AUC below threshold: {candidate_metrics['auc']:.4f} < {min_auc}"
            logger.warning(f"❌ Guardrail 1 FAILED: {reason}")
            return False, reason
        logger.info(f"✅ Guardrail 1 PASSED: AUC {candidate_metrics['auc']:.4f} >= {min_auc}")
        
        # Guardrail 2: Minimum Recall threshold
        min_recall = self.promotion_rules['min_recall']
        if candidate_metrics['recall'] < min_recall:
            reason = f"Recall below threshold: {candidate_metrics['recall']:.4f} < {min_recall}"
            logger.warning(f"❌ Guardrail 2 FAILED: {reason}")
            return False, reason
        logger.info(f"✅ Guardrail 2 PASSED: Recall {candidate_metrics['recall']:.4f} >= {min_recall}")
        
        # Guardrail 3: Added complexity must improve the primary metric.
        min_auc_gain = self.promotion_rules.get('min_auc_gain', 0.0)
        auc_diff = candidate_metrics['auc'] - baseline_metrics['auc']
        if auc_diff < min_auc_gain:
            reason = (
                f"Candidate does not improve baseline validation AUC: "
                f"difference {auc_diff:.4f} < required gain {min_auc_gain:.4f}"
            )
            logger.warning(f"❌ Guardrail 3 FAILED: {reason}")
            return False, reason
        logger.info(
            f"✅ Guardrail 3 PASSED: AUC difference {auc_diff:+.4f} "
            f">= required gain {min_auc_gain:.4f}"
        )
        
        # All guardrails passed
        reason = "All promotion guardrails passed"
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ PROMOTION DECISION: PROMOTE CANDIDATE MODEL")
        logger.info("=" * 80)
        
        return True, reason
    
    def save_comparison_report(self, comparison: Dict, should_promote: bool, reason: str):
        """
        Save model comparison report
        
        Args:
            comparison: Comparison results
            should_promote: Whether to promote candidate
            reason: Reason for promotion decision
        """
        report = {
            'comparison': comparison,
            'promotion_decision': {
                'should_promote': should_promote,
                'reason': reason,
                'promoted_model': 'candidate' if should_promote else 'baseline'
            },
            'promotion_rules': self.promotion_rules
        }
        
        report_dir = Path('artifacts/eval')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = report_dir / 'model_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✅ Comparison report saved to {json_path}")
        
        # Save Markdown report
        md_path = report_dir / 'model_comparison.md'
        self._save_markdown_report(md_path, comparison, should_promote, reason)
        logger.info(f"✅ Markdown report saved to {md_path}")

        champion_type = 'candidate' if should_promote else 'baseline'
        champion = {
            'model_type': champion_type,
            'model_version': f'{champion_type}_v1.0.0',
            'model_path': (
                'models/candidate/neural_network_v1.h5'
                if should_promote else 'models/baseline/logistic_regression_v1.pkl'
            ),
            'selection_dataset': 'validation',
            'reason': reason
        }
        champion_path = Path(self.config['models'].get('current_best_path', 'models/current_best.json'))
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        with open(champion_path, 'w') as f:
            json.dump(champion, f, indent=2)
        logger.info(f"✅ Champion manifest saved to {champion_path}")
    
    def _save_markdown_report(self, filepath: Path, comparison: Dict, should_promote: bool, reason: str):
        """
        Save comparison report in Markdown format
        """
        baseline = comparison['baseline']
        candidate = comparison['candidate']
        diffs = comparison['differences']
        
        with open(filepath, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write("## Validation Performance Used for Promotion\n\n")
            f.write("| Metric | Baseline | Candidate | Difference | Winner |\n")
            f.write("|--------|----------|-----------|------------|--------|\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                b_val = baseline[metric]
                c_val = candidate[metric]
                diff = diffs[metric]
                winner = "**Candidate**" if diff > 0 else "**Baseline**" if diff < 0 else "Tie"
                f.write(f"| {metric.upper()} | {b_val:.4f} | {c_val:.4f} | {diff:+.4f} | {winner} |\n")

            final_test = comparison.get('final_test', {})
            if final_test.get('baseline') and final_test.get('candidate'):
                f.write("\n## Final Untouched Test-Set Performance\n\n")
                f.write("| Metric | Baseline | Candidate |\n")
                f.write("|--------|----------|-----------|\n")
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    f.write(
                        f"| {metric.upper()} | {final_test['baseline'][metric]:.4f} | "
                        f"{final_test['candidate'][metric]:.4f} |\n"
                    )
            
            f.write("\n## Promotion Decision\n\n")
            f.write(f"**Decision**: {'✅ PROMOTE CANDIDATE' if should_promote else '❌ KEEP BASELINE'}\n\n")
            f.write(f"**Reason**: {reason}\n\n")
            
            f.write("## Promotion Guardrails\n\n")
            f.write(f"1. Minimum AUC: {self.promotion_rules['min_auc']}\n")
            f.write(f"2. Minimum Recall: {self.promotion_rules['min_recall']}\n")
            f.write(f"3. Minimum validation AUC gain: {self.promotion_rules.get('min_auc_gain', 0.0)}\n\n")
            
            if should_promote:
                f.write("## Recommended Action\n\n")
                f.write("1. Deploy candidate model to production\n")
                f.write("2. Monitor performance closely for first week\n")
                f.write("3. Keep baseline model as rollback option\n")
            else:
                f.write("## Recommended Action\n\n")
                f.write("1. Keep baseline model in production\n")
                f.write("2. Investigate why candidate failed guardrails\n")
                f.write("3. Retrain candidate with adjustments\n")


def main():
    """
    Main evaluation script
    """
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Load reports
    baseline_report, candidate_report = evaluator.load_evaluation_reports()
    
    # Compare models
    comparison = evaluator.compare_models(baseline_report, candidate_report)
    
    # Check promotion guardrails
    should_promote, reason = evaluator.check_promotion_guardrails(baseline_report, candidate_report)
    
    # Save comparison report
    evaluator.save_comparison_report(comparison, should_promote, reason)
    
    logger.info("\n✅ Evaluation completed successfully!")
    
    # Exit with appropriate code
    if should_promote:
        logger.info("🎉 Candidate model promoted!")
        exit(0)
    else:
        logger.info("⚠️  Candidate model not promoted")
        exit(1)


if __name__ == "__main__":
    main()
