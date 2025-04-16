# 🏡 California Housing Price Prediction

This project uses the **California Housing Dataset** to build a predictive model for median house prices using **Linear Regression**. The goal is to explore the relationships between housing features and price, and assess the model's predictive accuracy using evaluation metrics and visualizations.

---

## 📂 Project Overview

- **Dataset**: California Housing data from `sklearn.datasets`
- **Target Variable**: `PRICE` – Median house value (in $100,000s)
- **Model**: Linear Regression
- **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib

---

## 📊 Features & Visualizations

### Input Features:
| Feature     | Description                                  |
|-------------|----------------------------------------------|
| `MedInc`    | Median income in the block group             |
| `HouseAge`  | Median house age                             |
| `AveRooms`  | Average number of rooms per household        |
| `AveBedrms` | Average number of bedrooms per household     |
| `Population`| Block group population                       |
| `AveOccup`  | Average number of household occupants        |
| `Latitude`  | Latitude of the block                        |
| `Longitude` | Longitude of the block                       |

### Visualizations:
- **Pairplot**: Scatter plots of each feature against house prices
- **Correlation Heatmap**: Identify multicollinearity between variables
- **Prediction Plot**: Actual vs. Predicted house prices (Train & Test sets)

---

## 🧪 Model Evaluation

| Metric        | Training Set | Testing Set |
|---------------|--------------|-------------|
| MSE (Error)   | ~0.52        | ~0.56       |
| R² Score      | ~0.61        | ~0.58       |

- The model captures key trends but has limited capacity for complex patterns.
- Median income (`MedInc`) has the strongest positive correlation with price.
- Features like population and average occupancy show weak influence on price.

---

## 🚀 Key Steps

1. **Load Data**: Fetch dataset using `fetch_california_housing()`
2. **EDA**: Explore features, check distributions and correlations
3. **Split Data**: 80% training, 20% testing
4. **Train Model**: Fit Linear Regression on training data
5. **Evaluate**: Compute MSE & R², visualize predictions

---

## 📌 Takeaways

- Simple linear regression provides a good baseline but has limitations.
- Data shows some non-linear patterns that may benefit from more complex models.
- Ideal for those beginning with regression and real estate data exploration.

---

## 🔄 Potential Improvements

- Apply regularization (Ridge, Lasso)
- Explore tree-based methods (Random Forest, Gradient Boosting)
- Use log transformation or polynomial features
- Implement cross-validation for robust evaluation

---

## 🧠 Author’s Note

This project serves as an introductory regression analysis using a real-world dataset. It’s designed to be a simple, reproducible workflow that can be extended for deeper modeling and experimentation.

---

