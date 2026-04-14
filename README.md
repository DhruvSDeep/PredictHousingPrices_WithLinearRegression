# PredictHousingPrices_WithLinearRegression

A multivariate linear regression model that predicts **housing prices** from 14 property features. The project includes two parallel implementations side-by-side: a **from-scratch version** using NumPy (manual gradient descent and backpropagation) and an **scikit-learn version** for comparison. The sklearn model achieves an R² of ~0.8.

---

## Highlights

- **From-scratch gradient descent** — forward pass, MSE cost function, partial derivatives, and weight updates all implemented manually with NumPy. No ML library for the core learning loop.
- **Side-by-side sklearn comparison** — the same data is also fitted with `sklearn.linear_model.LinearRegression`, making it easy to compare learned weights, bias, and evaluation metrics between the manual and library approaches.
- **Full preprocessing pipeline** — manual one-hot encoding of categorical features, binary conversion of yes/no columns, and per-feature scaling to [0, 1].
- **Rich visualization** — cost convergence curve, predicted-vs-actual scatter plots (plus an ideal-case reference), Seaborn pairplots of key features vs. price, and a correlation heatmap.

---

## Dataset

The included `Housing_modified_Dataset.csv` contains **545 houses** with 13 columns:

| Feature | Type | Description |
|---|---|---|
| area | Numeric | Property area (sq ft) |
| bedrooms | Numeric | Number of bedrooms |
| bathrooms | Numeric | Number of bathrooms |
| stories | Numeric | Number of stories |
| parking | Numeric | Number of parking spots |
| mainroad | Binary | On a main road (yes/no) |
| guestroom | Binary | Has a guest room (yes/no) |
| basement | Binary | Has a basement (yes/no) |
| hotwaterheating | Binary | Has hot water heating (yes/no) |
| airconditioning | Binary | Has air conditioning (yes/no) |
| prefarea | Binary | In a preferred area (yes/no) |
| furnishingstatus | Categorical | furnished / semi-furnished / unfurnished |
| **price** | **Target** | **House price** |

---

## Project Structure

```
PredictHousingPrices_WithLinearRegression/
├── ProjMan.ipynb                  # Full pipeline: preprocessing, manual regression, sklearn regression, evaluation
├── Housing_modified_Dataset.csv   # 545-row housing dataset
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run

Open and run `ProjMan.ipynb` end-to-end. The notebook is organized in two main sections:

**Part 1 — Manual Implementation**
1. Load and preprocess the data (null filling, one-hot encoding, feature scaling, binary conversion).
2. Initialize random weights (14 features) and bias.
3. Run gradient descent for 3 000 iterations (learning rate 1 × 10⁻⁴).
4. Evaluate on a 10% held-out test set — prints MSE, R², learned weights, and plots.

**Part 2 — sklearn Implementation**
1. Split data with `train_test_split` (10% test, random state 4000).
2. Fit `LinearRegression` in one call.
3. Evaluate — prints MSE, R² (~0.8), learned weights, and comparison plots.

---

## How It Works

### Preprocessing

The raw dataset contains a mix of numeric, binary (yes/no), and categorical columns. The notebook transforms them into a uniform numeric feature matrix:

- **Binary features** — "yes"/"no" strings are mapped to 1/0.
- **Categorical feature** — `furnishingstatus` is one-hot encoded into three columns: `furnished`, `semifurnished`, `unfurnished`.
- **Feature scaling** — `area` and `price` are divided by 10 000; `bedrooms`, `bathrooms`, and `stories` are divided by 4, bringing all values into a comparable [0, 1] range for stable gradient descent.

### Manual Gradient Descent

The learning loop computes predictions as `Ŷ = Xw + b`, calculates MSE cost, derives the gradients `dw` and `db` analytically, and updates the weights and bias by subtracting `learning_rate × gradient`. This runs for 3 000 steps at a learning rate of 0.0001. The cost curve is plotted to confirm convergence.

### Evaluation & Visualization

Both implementations report MSE and R² on the test set and display the learned weight for each feature (allowing comparison of which features the model considers most predictive). The notebook includes predicted-vs-actual scatter plots and an ideal y = x reference, a Seaborn pairplot of selected features against price, and a full correlation heatmap.

---

## License

No license specified. Contact the repository owner for usage terms.
