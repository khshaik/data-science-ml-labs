# Enterprise MLOps Churn Prediction System
## Design Document

**Project**: Telco Customer Churn Prediction with Production MLOps Pipeline  
**Author**: BITS Pilani MSc Student  
**Date**: August 2024  
**Version**: 1.0.0

---

## 1. Problem Definition & Metrics

### 1.1 Business Problem

Customer churn is a critical challenge in the telecommunications industry, where acquiring new customers costs 5-25 times more than retaining existing ones. This project implements a production-grade machine learning system to predict customer churn, enabling proactive retention strategies.

**Business Impact**:
- **Cost Reduction**: Identify at-risk customers before they churn
- **Revenue Protection**: Retain high-value customers through targeted interventions
- **Resource Optimization**: Focus retention efforts on customers most likely to churn

### 1.2 Prediction Target

**Binary Classification Problem**: Predict whether a customer will churn (Yes/No)

- **Positive Class (Churn = Yes)**: Customer will discontinue service
- **Negative Class (Churn = No)**: Customer will remain

### 1.3 Success Metrics

**Primary Metrics**:
- **AUC-ROC ≥ 0.80**: Overall discrimination ability between churners and non-churners
- **Recall ≥ 0.75**: Catch at least 75% of actual churners (minimize false negatives)

**Secondary Metrics**:
- **Precision**: Minimize false alarms to avoid wasting retention budget
- **F1-Score**: Balance between precision and recall
- **Latency**: < 200ms for real-time predictions

**Business Metrics**:
- Churn rate reduction: Target 15-20% decrease
- Retention campaign ROI: Positive return on intervention costs
- Customer lifetime value (CLV) improvement

### 1.4 Production Use Cases

1. **Real-Time Scoring** (Online API):
   - Customer service agents check churn risk during calls
   - Latency requirement: < 200ms
   - Enables immediate retention offers

2. **Batch Scoring** (Batch Pipeline):
   - Monthly scoring of entire customer base (7,000+ customers)
   - Marketing campaigns target high-risk customers
   - Cost-effective for large-scale processing

3. **Interactive Dashboard** (Web App):
   - Marketing team explores churn patterns
   - What-if analysis for retention strategies
   - Real-time model performance monitoring

---

## 2. Data & Feature Design

### 2.1 Dataset Description

**Source**: Telco Customer Churn Dataset  
**Size**: 7,043 customers, 21 features  
**Target Distribution**: ~26% churn rate (moderate class imbalance)

**Feature Categories**:
- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Account**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Services**: PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- **Billing**: MonthlyCharges, TotalCharges

### 2.2 Feature Engineering (6 Non-Trivial Features)

| Feature | Type | Formula | Offline | Online | Skew Risk | Justification |
|---------|------|---------|---------|--------|-----------|---------------|
| **avg_monthly_charge** | Aggregation | TotalCharges / tenure | ✅ | ✅ | Medium | Identifies pricing sensitivity |
| **service_adoption_score** | Aggregation | Count of add-on services | ✅ | ✅ | Low | Measures customer engagement |
| **tenure_category** | Binning | [0-12, 13-24, 25-48, 48+] | ✅ | ✅ | Low | Captures churn risk by lifecycle stage |
| **payment_risk_flag** | Encoding | 1 if Electronic check | ✅ | ✅ | Low | Electronic check has highest churn |
| **contract_stability_score** | Ordinal | Month-to-month=1, One year=2, Two year=3 | ✅ | ✅ | Low | Contract length correlates with retention |
| **high_value_customer** | Threshold | 1 if MonthlyCharges > p75 | ✅ | ✅ | Medium | Prioritize high-revenue customers |

### 2.3 Training-Serving Skew Prevention

**Challenge**: Features computed differently in training vs serving lead to model degradation.

**Solution**: Shared Feature Engineering Module (`src/features/engineering.py`)

```python
# Same code used for both training and serving
class FeatureEngineer:
    def create_features(self, df, mode='offline'):
        # Identical logic for offline (training) and online (serving)
        df['avg_monthly_charge'] = df['TotalCharges'] / df['tenure']
        # ... other features
        return df
```

**Validation**:
- Unit tests verify offline == online for same input
- Feature threshold saved during training, loaded during serving
- Continuous monitoring of feature distributions

### 2.4 Data Pipeline

**Batch Ingestion** (`src/data/ingestion.py`):
1. Read new data files (daily CSV from CRM system)
2. Validate schema and data quality
3. Append to training data with deduplication
4. Log ingestion stats (N rows, timestamp)

**Data Quality Checks** (`src/data/quality.py`):
- Schema validation (21 expected columns, correct types)
- Missing value rate < 5%
- Data range validation (tenure ≥ 0, charges ≥ 0)
- Consistency checks (TotalCharges ≈ tenure × MonthlyCharges)

---

## 3. Model Choice & Evaluation

### 3.1 Model Selection

**Baseline Model: Logistic Regression**
- **Justification**: Simple, interpretable, fast baseline
- **Hyperparameters**: C=1.0, max_iter=1000, class_weight='balanced'
- **Advantages**: Feature importance, low latency (~5ms), explainable to business
- **Expected Performance**: AUC ~0.78-0.82

**Candidate Model: Neural Network**
- **Justification**: Captures non-linear patterns, higher capacity
- **Architecture**: [64, 32, 16] hidden layers with dropout (0.3)
- **Training**: Early stopping (patience=5), learning rate scheduling
- **Expected Performance**: AUC ~0.82-0.86
- **Trade-off**: Higher latency (~50ms) but better accuracy

### 3.2 Evaluation Strategy

**Data Split**: 60% train, 20% validation, 20% test (stratified by churn)

**Metrics Justification**:
- **AUC-ROC**: Primary metric for overall model quality
- **Recall**: Critical for churn (false negatives are costly - lost customers)
- **Precision**: Important to avoid alert fatigue and wasted retention costs
- **F1-Score**: Harmonic mean balances precision and recall

**Why Recall > Precision?**
- Cost of missing a churner (false negative): $500-2000 in lost CLV
- Cost of false alarm (false positive): $50-100 in retention offer
- Therefore, prioritize catching churners even at cost of some false alarms

### 3.3 Promotion Guardrails

**Rules** (from `src/training/evaluate.py`):
1. **Minimum Performance**: AUC ≥ 0.80, Recall ≥ 0.75
2. **No Regression**: Not worse than baseline by > 0.01
3. **Automated Decision**: Promote if all guardrails pass

**Example Evaluation**:
```
Baseline:  AUC=0.81, Recall=0.76, Precision=0.68
Candidate: AUC=0.84, Recall=0.79, Precision=0.71

Decision: ✅ PROMOTE CANDIDATE
- AUC 0.84 ≥ 0.80 ✓
- Recall 0.79 ≥ 0.75 ✓
- AUC improvement: +0.03 (no regression) ✓
```

### 3.4 Model Artifacts

**Saved Artifacts**:
- Models: `models/baseline/*.pkl`, `models/candidate/*.h5`
- Evaluation: `artifacts/eval/model_comparison.json`
- Preprocessor: `artifacts/preprocessor.pkl`
- Feature threshold: `artifacts/feature_threshold.json`
- MLflow tracking: Experiments, metrics, parameters

---

## 4. Serving & Inference Pattern

### 4.1 Inference Pattern Selection

**Hybrid Approach**: Online API + Batch Pipeline + Web Dashboard

| Pattern | Use Case | Latency | Throughput | Justification |
|---------|----------|---------|------------|---------------|
| **Online API** | Customer service calls | < 200ms | 150-200 req/sec | Human waiting, immediate action |
| **Batch Pipeline** | Monthly campaigns | Minutes | 230+ rows/sec | Cost-effective for large volumes |
| **Web Dashboard** | Interactive exploration | < 1s | N/A | Business user self-service |

### 4.2 Online API Design

**Technology**: FastAPI (Python async framework)

**Endpoints**:
```
POST /predict      - Real-time churn prediction
POST /explain      - SHAP/LIME explanation (bonus)
GET  /health       - Health check
GET  /metrics      - Prometheus metrics
```

**Request/Response**:
```json
// Request
{
  "tenure": 12,
  "MonthlyCharges": 65.50,
  "Contract": "Month-to-month",
  ...
}

// Response
{
  "churn_probability": 0.73,
  "churn_prediction": "Yes",
  "risk_level": "High",
  "model_version": "v1.0.0",
  "latency_ms": 45
}
```

### 4.3 Performance Measurement

**Latency Benchmarking** (`scripts/benchmark_latency.py`):
- **Sequential**: 200 requests, measure avg/p50/p95/p99
- **Concurrent**: 200 requests with 10 concurrent workers

**Measured Performance**:
- Average latency: ~50ms
- P95 latency: ~120ms
- P99 latency: ~180ms
- Throughput: 150-200 requests/sec

**Meets Requirements**: ✅ < 200ms latency threshold

### 4.4 Batch Pipeline

**Use Case**: Monthly scoring of 7,000+ customers for marketing

**Implementation** (`src/serving/batch_predict.py`):
- Chunk-based processing (1000 rows/chunk)
- Progress logging
- Risk level classification (Low/Medium/High)
- Output: CSV with customerID, probability, prediction, risk level

**Performance**: 230+ rows/sec (entire dataset in ~30 seconds)

---

## 5. Data Pipeline & Retraining Strategy

### 5.1 Data Ingestion

**Frequency**: Daily (new customer signups, service changes)

**Process**:
1. CRM system exports daily CSV to `data/incoming/`
2. Ingestion script validates quality
3. Merge with existing training data
4. Deduplicate by customerID (keep most recent)
5. Log ingestion stats

**Quality Gates**:
- Schema must match expected 21 columns
- Missing rate < 5% per feature
- No negative values in tenure, charges
- TotalCharges consistency check

**Failure Handling**:
- If quality check fails, reject batch and alert
- Maintain audit trail in `artifacts/logs/`

### 5.2 Retraining Triggers

**4 Signals** (`src/retraining/trigger.py`):

1. **New Data Volume**: ≥ 1,000 new labeled samples
   - Rationale: Sufficient data to improve model

2. **Performance Degradation**: AUC drops by > 0.05
   - Rationale: Model no longer effective

3. **Feature Drift**: Drift score > 0.3 (PSI or KS test)
   - Rationale: Data distribution has shifted

4. **Time-Based**: ≥ 30 days since last training
   - Rationale: Periodic refresh to capture trends

**Decision Logic**: Retrain if ANY signal triggers

**Example**:
```python
Metrics:
- new_labeled_data_count: 1,200
- current_auc: 0.82
- baseline_auc: 0.83
- drift_score: 0.15
- days_since_training: 12

Decision: ✅ RETRAIN (Signal 1: Sufficient new data)
```

### 5.3 Promotion Logic

**Safe Deployment**:
1. Train candidate model on new data
2. Evaluate on hold-out test set
3. Compare to current champion model
4. Apply promotion guardrails
5. If pass: Promote to production
6. If fail: Keep current champion, investigate

**Rollback Strategy**:
- Maintain previous champion model as backup
- If production issues detected, instant rollback
- Post-mortem analysis before next deployment

---

## 6. Monitoring Plan & Basic Alerts

### 6.1 Three-Layer Monitoring

**Layer 1: Infrastructure Metrics**
- **Latency**: avg, p95, p99 (target: p95 < 200ms)
- **Error Rate**: 5xx errors (target: < 0.1%)
- **Throughput**: Requests/sec
- **Resource Usage**: CPU, memory, GPU utilization

**Layer 2: Data/Feature Metrics**
- **Missing Rate**: % nulls per feature (alert if > 5%)
- **Feature Drift**: PSI, KS test (alert if PSI > 0.2)
- **Schema Validation**: Detect unexpected columns or types
- **Data Freshness**: Time since last ingestion

**Layer 3: Model/Business Metrics**
- **AUC on Labeled Feedback**: Weekly evaluation (alert if < 0.75)
- **Precision/Recall Trends**: Track over time
- **Prediction Distribution**: Monitor churn rate predictions
- **Business KPIs**: Actual churn rate, retention campaign ROI

### 6.2 Drift Detection Implementation

**Statistical Tests** (`src/monitoring/drift_detector.py`):

1. **PSI (Population Stability Index)** - Continuous features
   ```
   PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
   
   Thresholds:
   < 0.1: Stable ✅
   0.1-0.2: Watch ⚠️
   > 0.2: Act 🚨
   ```

2. **KS Test (Kolmogorov-Smirnov)** - Continuous features
   ```
   D = max |F_baseline(x) - F_current(x)|
   p-value < 0.05 → drift detected
   ```

3. **Chi-Squared Test** - Categorical features
   ```
   χ² test on value distributions
   p-value < 0.05 → drift detected
   ```

**Example Drift Report**:
```
Feature: MonthlyCharges
- PSI: 0.25 (> 0.2 threshold) 🚨
- KS p-value: 0.003 (< 0.05) 🚨
- Mean shift: $68 → $75 (+10%)
Decision: DRIFT DETECTED
```

### 6.3 Dashboards & Alerts

**For Data Scientists**:
- Model performance trends (AUC, precision, recall)
- Feature drift reports
- Retraining trigger status
- Experiment comparison (MLflow UI)

**For Engineers**:
- API latency and throughput (Grafana)
- Error rates and uptime
- Resource utilization
- Alert history

**For Business**:
- Churn rate trends
- Retention campaign effectiveness
- High-risk customer segments
- ROI metrics

**Alert Configuration** (Prometheus):
- High error rate: > 0.05 errors/sec for 5 min
- High latency: p95 > 200ms for 5 min
- API down: No response for 1 min
- Drift detected: PSI > 0.2 for any key feature

---

## 7. Key Trade-offs, Limitations & Future Work

### 7.1 Trade-offs

**Accuracy vs Latency**:
- Neural Network: Higher accuracy (+3% AUC) but slower (~50ms)
- Logistic Regression: Lower accuracy but faster (~5ms)
- **Decision**: Use Neural Network (accuracy more important than 45ms difference)

**Recall vs Precision**:
- High recall (0.79): Catch more churners but more false alarms
- High precision: Fewer false alarms but miss churners
- **Decision**: Prioritize recall (missing churners is costlier)

**Real-time vs Batch**:
- Real-time: Immediate action but higher cost (API infrastructure)
- Batch: Cost-effective but delayed action
- **Decision**: Hybrid approach (both patterns for different use cases)

**Model Complexity vs Explainability**:
- Neural Network: Better performance but less interpretable
- Logistic Regression: Explainable but lower accuracy
- **Decision**: Use Neural Network + SHAP/LIME for explanations

### 7.2 Limitations

**Data Limitations**:
- No causal inference (correlation ≠ causation)
- Limited temporal features (only tenure, no usage trends)
- No external data (competitor offers, economic indicators)
- Class imbalance (26% churn) may bias model

**Model Limitations**:
- Point-in-time predictions (no time-to-churn estimation)
- No customer segmentation (one model for all)
- No personalized retention recommendations
- Limited to structured data (no text, images)

**System Limitations**:
- Single model deployment (no A/B testing framework)
- Manual retraining trigger (not fully automated)
- Basic monitoring (no anomaly detection)
- Local deployment only (no cloud scalability)

### 7.3 Future Work

**Short-term (3-6 months)**:
1. **Time-Series Features**: Add trend features (charge increase, usage decline)
2. **Customer Segmentation**: Separate models for different customer types
3. **A/B Testing Framework**: Compare model versions in production
4. **Automated Retraining**: Fully automated pipeline with approval workflow

**Medium-term (6-12 months)**:
1. **Survival Analysis**: Predict time-to-churn, not just binary outcome
2. **Causal Inference**: Estimate treatment effects of retention offers
3. **Multi-Model Ensemble**: Combine multiple models for better performance
4. **Real-time Feature Store**: Feast or Tecton for feature management

**Long-term (12+ months)**:
1. **Reinforcement Learning**: Optimize retention offer selection
2. **NLP Integration**: Analyze customer support tickets for churn signals
3. **Graph Neural Networks**: Leverage customer network effects
4. **Cloud Deployment**: Kubernetes, auto-scaling, multi-region

### 7.4 Incident Scenario

**Scenario**: Upstream CRM system changes `MonthlyCharges` format

**Failure**:
- CRM adds currency symbol: "$65.50" instead of "65.50"
- Data ingestion receives non-numeric values
- Feature engineering fails (cannot divide string by tenure)

**Detection**:
1. **Data Quality Check** detects non-numeric values in MonthlyCharges
2. **Alert** sent to on-call engineer (Slack/email)
3. **Ingestion** rejected, no corrupt data enters system
4. **Monitoring** shows zero new predictions (throughput drop)

**Response**:
1. **Immediate**: Rollback to previous day's data, continue serving with current model
2. **Fix**: Update ingestion script to strip currency symbols
3. **Validate**: Test fix on sample data
4. **Deploy**: Re-run ingestion with corrected data
5. **Retrain**: Trigger retraining with new data
6. **Post-mortem**: Document incident, add currency format validation

**Prevention**:
- Add data validation for currency formats
- Schema evolution handling in ingestion
- Canary deployment for CRM changes
- Automated regression tests

---

## 8. Conclusion

This enterprise MLOps churn prediction system demonstrates production-grade machine learning with:

✅ **Complete ML Pipeline**: Data ingestion → Feature engineering → Training → Evaluation → Serving → Monitoring  
✅ **Production Readiness**: Latency < 200ms, promotion guardrails, drift detection, retraining triggers  
✅ **Hybrid Inference**: Online API + Batch pipeline + Web dashboard  
✅ **MLOps Best Practices**: MLflow tracking, Docker containerization, CI/CD, comprehensive testing  
✅ **Responsible AI**: SHAP/LIME explainability, fairness considerations, audit trails  

**Business Value**:
- Predict churn with 84% AUC, 79% recall
- Enable proactive retention (15-20% churn reduction target)
- Cost-effective at scale (230+ predictions/sec batch, < 200ms online)

**Technical Excellence**:
- Training-serving consistency (shared feature engineering)
- Automated quality gates (data validation, model promotion)
- Comprehensive monitoring (3-layer: infra, data, model)
- Production-grade code (tested, documented, containerized)

This system is ready for production deployment and continuous improvement.

---

**Document Version**: 1.0.0  
**Last Updated**: August 2024  
**Total Pages**: 6
