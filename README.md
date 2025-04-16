# 🏡 California Housing Price Prediction using Linear Regression

This project demonstrates a machine learning workflow to predict housing prices using the California Housing Dataset. The approach focuses on using a Linear Regression model with data from the `sklearn.datasets` module.

---

## 📌 Project Workflow

1. **Load the Dataset**  
   The California Housing dataset is loaded and converted into a Pandas DataFrame. The target variable is the median house value, labeled as `PRICE`.

2. **Explore and Understand the Data**  
   - Preview the dataset structure and data types  
   - Examine basic statistics (mean, median, etc.)  
   - Identify any missing values  

3. **Visualize the Data**  
   - Use scatter plots to visualize relationships between features and the target (`PRICE`)  
   - Plot a heatmap to explore correlations between features  

4. **Data Preprocessing**  
   - Confirm no missing values exist  
   - Ensure features and target are separated  
   - Split the dataset into training and testing sets (80/20 split)  

5. **Model Training**  
   - Apply a Linear Regression model on the training data  
   - Fit the model and learn the weights for each feature  

6. **Model Evaluation**  
   - Evaluate the model using MSE (Mean Squared Error) and R² (R-squared) on both training and testing data  
   - Key metrics observed:  
     - Training MSE: ~0.52  
     - Testing MSE: ~0.56  
     - Training R²: ~0.61  
     - Testing R²: ~0.58  

7. **Visualize Predictions**  
   - Compare actual vs predicted prices for both training and test datasets  
   - Include an "ideal fit" reference line to evaluate prediction alignment  

---

## 📊 Features & Visualizations

- **Features in the dataset**:
  - `MedInc`: Median income in the block group
  - `HouseAge`: Median house age in the block group
  - `AveRooms`: Average number of rooms
  - `AveBedrms`: Average number of bedrooms
  - `Population`: Block group population
  - `AveOccup`: Average number of occupants per household
  - `Latitude` and `Longitude`: Geographical location

- **Visualizations included**:
  - Pairplots for each feature against the target variable
  - Correlation heatmap of all numerical variables
  - Scatter plot comparing actual vs predicted house prices

---

## 🔍 Sample Insights

- **Strongest positive correlation** with housing price: Median Income (`MedInc`)
- **Weak or negative correlation**: Population, Average Occupancy
- Linear regression captures basic patterns, but might not account well for non-linear relationships
- Model performance shows moderate predictive power, but leaves room for improvement with more advanced methods

---

## 📁 Dataset Information

- **Source**: California Housing Dataset from `sklearn.datasets`
- **Size**: 20,640 rows × 9 columns
- **License**: Public domain

---

## ✅ Next Steps

- Perform feature engineering (e.g., interaction terms, normalization)
- Try other regression algorithms (e.g., Ridge, Lasso, Random Forest, XGBoost)
- Add cross-validation for more robust model evaluation
- Include residual analysis and error distribution visualization

---

## 🧠 Author's Note

This is a foundational regression example ideal for beginners exploring housing data or testing regression workflows in Python with Scikit-learn. It’s also a great base project to extend into more complex modeling or deployment.

---
