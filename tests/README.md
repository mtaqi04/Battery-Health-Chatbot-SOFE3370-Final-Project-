# Testing Documentation

This directory contains comprehensive end-to-end tests for the Battery Health Chatbot pipeline.

## Test Files

- **`test_pipeline_e2e.py`** - Main end-to-end test suite
- **`validation_checklist.md`** - Manual validation checklist

## Running Tests

### Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Execute Tests

Run the complete test suite:
```bash
python tests/test_pipeline_e2e.py
```

Or from the project root:
```bash
cd tests
python test_pipeline_e2e.py
```

### Expected Output

The test suite will:
1. Validate data preprocessing
2. Test model loading
3. Verify predictions
4. Check classification logic
5. Test consistency across multiple runs
6. Validate chatbot integration
7. Test edge cases

Results are saved to `reports/test_logs/pipeline_test_YYYYMMDD_HHMMSS.json`

## Test Coverage

### Test Categories

1. **Data Preprocessing Validation**
   - Data file existence and structure
   - Feature completeness (U1-U21)
   - Data quality checks
   - SOH range validation

2. **Model Loading & Validation**
   - Model file existence
   - Model structure verification
   - Prediction capability

3. **Prediction Accuracy**
   - Single prediction validation
   - Batch prediction validation
   - Accuracy metrics (MAE, MSE, RMSE)

4. **Classification Logic**
   - Multiple threshold testing
   - Classification correctness
   - Batch classification distribution

5. **Consistency Testing**
   - Same sample multiple runs
   - Different samples
   - Model persistence

6. **Chatbot Integration**
   - Input parsing (various formats)
   - Feature extraction
   - Prediction integration

7. **Edge Cases & Error Handling**
   - Invalid inputs
   - Extreme values
   - Boundary conditions

## Performance Benchmarks

| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| Model Loading Time | < 1s | < 2s |
| Single Prediction Time | < 0.1s | < 0.5s |
| Batch Prediction (670 samples) | < 5s | < 10s |
| Prediction Consistency (std) | < 1e-10 | < 1e-8 |
| Sample MAE | < 0.1 | < 0.15 |
| Sample RMSE | < 0.15 | < 0.2 |

## Success Criteria

All tests should pass with:
- 100% test pass rate
- Consistent results across multiple runs
- All predictions in valid range [0, 1]
- Classification logic correct for all thresholds
- No errors in error handling tests

## Test Results

Test results are automatically saved to:
```
reports/test_logs/pipeline_test_YYYYMMDD_HHMMSS.json
```

Each test run includes:
- Timestamp
- Individual test results
- Error logs (if any)
- Summary statistics

## Multiple Run Validation

For complete validation, run the test suite multiple times:

```bash
# Run 5 times for consistency validation
for i in {1..5}; do
    echo "Run $i:"
    python tests/test_pipeline_e2e.py
    echo ""
done
```

On Windows PowerShell:
```powershell
for ($i=1; $i -le 5; $i++) {
    Write-Host "Run $i:"
    python tests/test_pipeline_e2e.py
    Write-Host ""
}
```

Expected: 100% consistency across all runs.

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `models/soh_linear_model.pkl` exists
   - Check file permissions

2. **Data file not found**
   - Ensure `data/cleaned_pulsebat.csv` exists
   - Verify file path

3. **Import errors**
   - Ensure you're running from project root
   - Check Python path configuration

4. **Inconsistent results**
   - Check for random seed issues
   - Verify model is deterministic

## Additional Resources

- Validation Checklist: `tests/validation_checklist.md`
- Model Metrics: `reports/threshold_metrics.csv`
- Confusion Matrix: `reports/confusion_matrix.csv`

---

