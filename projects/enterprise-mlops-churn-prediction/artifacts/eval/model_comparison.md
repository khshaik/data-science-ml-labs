# Model Comparison Report

## Validation Performance Used for Promotion

| Metric | Baseline | Candidate | Difference | Winner |
|--------|----------|-----------|------------|--------|
| ACCURACY | 0.7488 | 0.7353 | -0.0135 | **Baseline** |
| PRECISION | 0.5174 | 0.5009 | -0.0165 | **Baseline** |
| RECALL | 0.7941 | 0.7674 | -0.0267 | **Baseline** |
| F1 | 0.6266 | 0.6061 | -0.0205 | **Baseline** |
| AUC | 0.8354 | 0.8300 | -0.0054 | **Baseline** |

## Final Untouched Test-Set Performance

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| ACCURACY | 0.7367 | 0.7324 |
| PRECISION | 0.5026 | 0.4975 |
| RECALL | 0.7807 | 0.7914 |
| F1 | 0.6115 | 0.6109 |
| AUC | 0.8429 | 0.8364 |

## Promotion Decision

**Decision**: ❌ KEEP BASELINE

**Reason**: Candidate does not improve baseline validation AUC: difference -0.0054 < required gain 0.0000

## Promotion Guardrails

1. Minimum AUC: 0.8
2. Minimum Recall: 0.75
3. Minimum validation AUC gain: 0.0

## Recommended Action

1. Keep baseline model in production
2. Investigate why candidate failed guardrails
3. Retrain candidate with adjustments
