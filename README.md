# 🏡 California Housing Price Prediction

This project demonstrates the use of Linear Regression to predict housing prices in California based on various features, using the **California Housing Dataset**. The project has been enhanced with additional steps for feature engineering, advanced model evaluation, and residual analysis to improve the performance and interpretability of the model.

---

## 📂 Project Overview

- **Dataset**: California Housing data from `sklearn.datasets`
- **Target Variable**: `PRICE` – Median house value (in $100,000s)
- **Model**: Linear Regression
- **Tech Stack**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib

In this project, we use the California Housing dataset to train a Linear Regression model for predicting housing prices. The dataset includes various features such as the average income, house age, average number of rooms, and more. The main goal is to build a model that predicts the housing prices based on these features and evaluate its performance using various metrics.

The project has been extended to include:
- Feature Engineering (creating new features)
- Model evaluation using advanced techniques
- Residual analysis to detect patterns or biases in the model
- Implementing cross-validation for a more robust model evaluation

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

### Data Exploration: 
  - The dataset contains the following features: Median Income (`MedInc`), House Age (`HouseAge`), Average Rooms (`AveRooms`), Average Bedrooms (`AveBedrms`), Population (`Population`), Average Occupancy (`AveOccup`), Latitude (`Latitude`), Longitude (`Longitude`), and the Target Variable `PRICE` (Median House Price).
  - We performed exploratory data analysis (EDA) to understand the distributions of these features, visualize their relationships with the target variable, and detect any correlations.

- **Correlation Matrix**: 
  - A heatmap was used to display the correlation between features, helping us identify which features are most strongly correlated with the target variable.

- **Feature Engineering**: 
  - We created new features, including `AvgOccupPerRoom`, which combines `AveOccup` and `AveRooms`, to capture additional information about the occupancy rate per room. This can improve the model's predictive power.

- **Visualization of Actual vs Predicted Prices**: 
  - A scatter plot of predicted vs actual housing prices was used to visually assess the accuracy of the model's predictions.
---

## 🧪 Model Evaluation

| Metric        | Training Set | Testing Set |
|---------------|--------------|-------------|
| MSE (Error)   | ~0.52        | ~0.56       |
| R² Score      | ~0.61        | ~0.58       |

- The model captures key trends but has limited capacity for complex patterns.
- Median income (`MedInc`) has the strongest positive correlation with price.
- Features like population and average occupancy show weak influence on price.

### Sample Insights

- **High Correlation Between `MedInc` and `PRICE`**:
  - Median income (`MedInc`) has the highest correlation with the target variable, suggesting that income plays a significant role in determining house prices.

- **Feature Engineering Impact**:
  - The addition of the `AvgOccupPerRoom` feature helped the model capture nuances related to the occupancy rate, improving predictions.

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
# California Housing Price Prediction





## Additional Enhancements

1. **Data Preprocessing**:
   - Checked for missing values (none present).
   - Displayed summary statistics to understand the range and distribution of the data.

2. **Model Improvements**:
   - Implemented **Advanced Regression Models** such as **Ridge** and **Lasso Regression** to improve generalization and reduce overfitting by applying regularization.
   - **Hyperparameter tuning** using `GridSearchCV` to find the optimal settings for the models.

3. **Residual Analysis**:
   - Visualized residuals to check if the model assumptions hold. We plotted the residuals and checked for any patterns (non-random distribution could indicate problems in the model).

4. **Cross-Validation**:
   - Used **K-fold cross-validation** to assess the performance of the model on different subsets of the data, ensuring it performs well on unseen data.

## Model Evaluation

After training the linear regression model, the following metrics were calculated:

- **Mean Squared Error (MSE)**: A measure of the average squared difference between the predicted and actual values.
- **R-Squared (R²)**: A statistical measure that indicates the proportion of the variance in the target variable that is explained by the model.

The model evaluation showed the following results:
- **Training MSE**: 0.5179
- **Testing MSE**: 0.5559
- **Training R²**: 0.6126
- **Testing R²**: 0.5758

## Conclusion

This project demonstrates a basic machine learning pipeline for housing price prediction using the California Housing dataset. By applying feature engineering, advanced modeling techniques, and rigorous model evaluation, we have improved the model's accuracy and reliability. The use of residual analysis and cross-validation helped ensure that the model generalizes well to new data.

Future steps could include:
- Implementing additional machine learning models like Random Forests or Gradient Boosting for better accuracy.
- Deploying the model as a web service for real-time predictions.

## Dependencies

The following Python libraries are required to run the code:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

