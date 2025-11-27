# 3 Model Design & Architecture

## 3.1 Model Architecture

A linear regression model was implemented as the starting method for prediction for a batteries state of health (SOH). The dataset consisted of 21 voltage based features, and the target variable was SOH. Linear regression was used in the model due to its interoperability, lower computational cost and overall effectiveness of analyzing and determining relationships between input voltage and SOH. 

The model follows the general form:

ŷ = 0 + i=121ixi

### Model Characteristics

| Attribute | Value |
|-----------|-------|
| Dataset | cleaned_pulsebat.csv |
| Features | U1–U21 |
| Target Variable | SOH |
| Intercept | 4.8639 |
| Number of Coefficients | 21 |
| Train/Test Split | 80% / 20% |
| Random State | 42 |
| Scaling | None required |
| Saved Model | soh_linear_model.pkl |

## 3.2 Training Procedure

The cleaned_pulsebat dataset was loaded and split into both training and testing subsets using a split ratio of 80/20. A regression model from the scikit-learn library was fitted within the training data without normalization as voltage values were consistent and decently scaled. The models training and eval were performed in model_training.ipynb, where it was exported and passed to a new directory to be stored for later usage.

## 3.3 Performance Metrics

### 3.3.1 Regression performance

The model was evaluated on the test set using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE). Results are summarized in the following table:

| Metric | Value |
|--------|-------|
| R² Score | 0.656088356 |
| Mean Squared Error | 0.001498344 |
| Mean Absolute Error | 0.030275494 |

**Interpretation:**

The model explains that approximately 66% of SOH variance and that the MAE indicated that predictions deviate from true SOH values by approximately 3% on average. The model establishes an overall baseline that demonstrates a reliable predictive capability given linear assumptions.

### 3.3.2 Performance Visualization

**Figure # - Predicted vs actual SOH plot**

A predicted vs actual SOH plot was generated to assess alignment between model predictions and true values. The plot shows that most predictions lie fairly near to the reference line, indicating a strong correlation between prediction and actual values.

**Figure # - Residuals vs Predicted SOH plot**

It is also observable that low SOH samples are slightly underestimated while high SOH values are slightly overestimated. Residuals due however appear to be randomly distributed, suggesting that there is an insignificant system error present.

## 3.4 Classification Threshold Evaluation

The predicted SOH values were converted into two classes using a threshold of 0.60 as the divider.

- **Healthy:** SOH ≥ 0.60
- **Has a Problem:** SOH < 0.60

The actual SOH labels were converted using the threshold. Accuracy, Precision, Recall, and F1 Scores were computed, and a confusion matrix was created.

**Figure # - Confusion Matrix**

The confusion matrix indicates misclassifications occur for SOH values near the threshold boundary (approximately 0.6 by a factor of 0.05).

## 3.5 Analysis and Discussion

The regression model provides a strong interpretable baseline for SOH with an R2 of approximately 0.66, with low error values. The model captures the variability within the SOH through the use of the voltage features. The visual analysis then confirms a generally linear relationship with little to no deviations. The threshold based classification demonstrates performance but highlights the ambiguity between some of the cases. The cases represent the misclassification, due to the narrow margin between classes.

### Limitations

- Linear Regression cannot model nonlinear relationships
- Voltage-only features may not completely capture degradation
- Threshold-based classification is perceptible to small regression errors

### Opportunities for Improvement

- Evaluate Polynomial Regression to capture any nonlinear trends/ relationships
- List 2 more
