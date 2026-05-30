# Lasso Regression

A Machine Learning project that demonstrates how **Lasso Regression** helps reduce overfitting and perform **automatic feature selection** using **L1 Regularization**.

This project explores how adding an L1 penalty to the cost function can shrink some coefficients exactly to zero, making the model simpler and more interpretable.

---

## Overview

Traditional Linear Regression models may overfit when:

- The dataset contains many features
- Features are highly correlated
- Noise exists in the training data

Lasso Regression addresses these issues by penalizing large coefficients and automatically removing less important features.

This project:

- Implements Lasso Regression using Scikit-learn
- Compares Lasso Regression with Linear Regression
- Demonstrates feature selection
- Explores coefficient shrinkage
- Visualizes the impact of different alpha values

---

## Machine Learning Concepts Covered

### 1. Lasso Regression

Lasso Regression is a regularized version of Linear Regression that adds an L1 penalty term.

Cost Function:

```text
J(θ) = Σ(yi - ŷi)² + λΣ|θj|
```

Where:

- `Σ(yi - ŷi)²` → Residual Sum of Squares
- `λ` → Regularization parameter
- `|θj|` → Absolute value of model coefficients

---

### 2. L1 Regularization

L1 Regularization helps:

- Reduce overfitting
- Improve model generalization
- Simplify models
- Perform automatic feature selection

---

### 3. Feature Selection

Unlike Ridge Regression, Lasso Regression can reduce some coefficients exactly to zero, effectively removing unimportant features.

---

### 4. Bias-Variance Tradeoff

The project demonstrates how regularization balances:

- Model complexity
- Prediction accuracy
- Generalization performance

---

## Project Workflow

### Step 1 — Import Libraries

The notebook imports:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

### Step 2 — Load Dataset

Dataset is loaded using Pandas.

---

### Step 3 — Data Preprocessing

The project prepares:

- Features (`X`)
- Target variable (`y`)

and performs train-test splitting.

---

### Step 4 — Train Linear Regression Model

A baseline Linear Regression model is trained for comparison.

---

### Step 5 — Train Lasso Regression Model

Lasso Regression is implemented using:

```python
from sklearn.linear_model import Lasso
```

---

### Step 6 — Experiment with Alpha Values

Different regularization strengths (`alpha`) are tested to observe their impact on:

- Coefficients
- Feature selection
- Model complexity
- Predictions

---

### Step 7 — Visualization and Analysis

The notebook visualizes:

- Coefficient shrinkage
- Feature elimination
- Regularization effects
- Model performance

---

## Key Insights

The project demonstrates:

- Why overfitting occurs
- How L1 Regularization works
- Automatic feature selection using Lasso
- Effect of alpha values on coefficients
- Difference between Linear, Ridge, and Lasso Regression

---

## License

This project is open-source and available under the MIT License.
