# 👥 Project Contributions & Responsibilities

This section outlines the distribution of tasks and technical responsibilities between the team members.

## 🤝 Collaborative Team Work (Shared Effort)
*The following tasks were executed through peer programming and team collaboration.*

* **3. Cleaning & Imputation**
    * Defining and justifying imputation techniques (Median, Mode, Sentinel).
    * Outlier detection and treatment using Winsorizing strategies.
* **4. Encoding Categorical Variables**
    * Implemented **One-Hot Encoding**, **Target/Mean Encoding** (with smoothing), and **Frequency Encoding**.
    * Applied precautions to avoid target leakage within Cross-Validation.
* **5. Feature Scaling**
    * Applied `StandardScaler` and `MinMaxScaler` based on model sensitivity (e.g., for KNN and SVR).
* **6. Feature Engineering (Mandatory Suite)**
    * **User-level:** Total orders, basket size, and reorder ratios.
    * **Product-level:** Popularity trends and reorder rates.
    * **Interaction & Non-linear:** User-Product interactions and Log transformations.
* **7. Dimensionality & Collinearity**
    * Identified multicollinearity using **VIF (Variance Inflation Factor)** and applied regularization.
* **8. Imbalanced Data Handling**
    * Experimented with **Class Weights** and **Sampling (SMOTE)** for classification tasks.

---

---

## 👨‍💻 Individual Contributions: **Zaid**
*Focused on the core infrastructure, data integrity, and temporal validation.*

* **1. Data Ingestion & Memory Management**
    * Developed an automated Python-based pipeline for merging multiple relational files (orders, products, aisles, departments).
    * Implemented memory-savvy techniques to process and join over **10.6 Million rows** efficiently.
* **2. Comprehensive Exploratory Data Analysis (EDA)**
    * **Missing Value Analysis:** Visualizing and quantifying data gaps.
    * **Distribution Analysis:** Plotting histograms and density plots for numeric features and the target variable.
    * **Cardinality & Seasonality:** Analyzing categorical top-K frequencies and purchase patterns (Hour-of-day, Day-of-week, Monthly cycles).
* **3. Time-Aware Validation Strategy**
    * Designed and implemented the **Chronological Split** (Time-aware splitting).
    * Ensured models are trained on historical data and validated on subsequent future orders to prevent **Temporal Data Leakage**.

---




# 🎯 Task A: Classification Implementation (Zaid's Core Work)

This task focuses on building a robust classification pipeline to predict customer behavior. The following work, from model selection to advanced evaluation, represents Zaid's individual contribution.

---

## 1. Classification Models Implemented
We explored a variety of architectures, ensuring a balance between baseline linear models and advanced ensembles:

* **Linear & Distance-Based:**
    * **Logistic Regression:** Optimized with L1/L2 regularization and class weight adjustments.
    * **K-Nearest Neighbors (KNN):** Implemented for local pattern recognition.
    * **Support Vector Machine (SVM):** Applied both Linear and Kernel (RBF) versions, while addressing computational limits through strategic sampling.
* **Tree-Based Ensembles:**
    * **Decision Tree & Random Forest Classifiers.**
    * **Gradient Boosting (XGBoost / LightGBM):** Required for high-performance categorical handling.

---

## 2. Advanced Evaluation Metrics (Mandatory Comparison)
To ensure the models are reliable and handle data imbalances effectively, I computed and compared the following suite of metrics for all models:

* **Standard Performance Metrics:**
    * **Accuracy, Precision, Recall, and F1-score:** To provide a foundational view of classification quality.
* **Discriminative Ability:**
    * **ROC Curve + AUC:** To evaluate the model's ability to distinguish between classes across all thresholds.
    * **Precision-Recall Curve + Average Precision (AP):** Crucial for this dataset to assess performance specifically on imbalanced target classes.
* **Error & Reliability Analysis:**
    * **Confusion Matrix:** Generated using both **normalized** and **raw counts** to pinpoint specific misclassification types.
    * **Calibration Curves (Reliability Diagrams):** Used for probability calibration to ensure that predicted probabilities reflect real-world likelihoods.
    * **MCC (Matthews Correlation Coefficient):** Included as an optional robust metric to provide a balanced quality score even for imbalanced classes.
 

# 📉 Task B: Regression Implementation (Hamza's Core Work)

This task involved predicting the continuous target variable (days_since_prior_order). The following modeling and statistical evaluation represent Hamza's individual contribution to the project.

---

## 1. Regression Models Implemented
We explored several architectures to capture the purchasing cycles of customers:

* **Linear & Regularized Models:**
    * **Ordinary Least Squares (OLS) / Linear Regression.**
    * **Regularized Variants:** Lasso (L1), Ridge (L2), and Elastic Net to handle feature sparsity and prevent overfitting.
* **Non-Linear & Distance-Based:**
    * **Support Vector Regressor (SVR):** Implemented with both Linear and RBF kernels.
    * **K-Nearest Neighbors (KNN) Regressor:** Tested with various 'k' values and distance metrics.
* **Tree-Based Ensembles:**
    * **Decision Tree & Random Forest Regressors.**
    * **Gradient Boosting Regressor (LightGBM/XGBoost):** *Required* - Optimized for large-scale data to achieve the best performance.

---

## 2. Quantitative Evaluation & Diagnostics
To ensure the statistical validity of the regression results, Hamza computed and analyzed the following:

* **Standard Error Metrics:**
    * **MAE (Mean Absolute Error), MSE (Mean Squared Error), and RMSE (Root Mean Squared Error).**
* **Goodness of Fit:**
    * **R² (Coefficient of Determination)** and **Adjusted R²** to measure the explained variance.
* **Statistical Diagnostics:**
    * **Residual Plots:** To visualize prediction errors.
    * **Heteroscedasticity Tests:** Specifically the **Breusch-Pagan test** to check for non-constant variance.
    * **Q-Q Plots:** To assess the normality of residuals.



 # 📊 Project Progress & Contribution Summary

### **Final status : 60% Completed**
---

## 📈 Contribution Breakdown
| Member | Role | Contribution % |
| :--- | :--- | :--- |
| **Zaid** | Infrastructure, EDA & Classification Lead | **33%** |
| **Hamza** | Regression & Statistical Diagnostics Lead | **27%** |
| **Total** | | **60%** |

---

## 👨‍💻 Detailed Individual Contributions: **Zaid (33%)**
*Primary focus on the backbone of the project and the behavioral classification engine.*

* **Data Ingestion & Memory Management:** Built an automated Python pipeline to handle 10.6M+ records efficiently.
* **Time-Aware Splitting:** Implemented a **Chronological Split** to ensure real-world predictive validity and prevent data leakage.
* **Exploratory Data Analysis (EDA):** Full missing value analysis, distribution studies, and temporal seasonality.
* **Task A (Classification):**
    * **Models:** Logistic Regression, KNN, SVM (with sampling), Random Forest, and **LightGBM/XGBoost**.
    * **Metrics:** ROC-AUC, **Precision-Recall Curves**, Confusion Matrices, **Calibration Diagrams**, and MCC.

---

## 👨‍💻 Detailed Individual Contributions: **Hamza (27%)**
*Primary focus on continuous prediction and statistical rigor for the regression task.*

* **Task B (Regression):**
    * **Models:** OLS, Lasso, Ridge, Elastic Net, SVR, KNN Regressor, and **LightGBM/XGBoost**.
* **Quantitative Diagnostics:**
    * **Metrics:** MAE, MSE, RMSE, and R² / Adjusted R².
    * **Statistical Tests:** Residual plots, **Breusch-Pagan test** for Heteroscedasticity, and Q-Q plots for normality.

---

## 🤝 Collaborative Team Work (Shared Foundation)
*Shared responsibilities in the data preparation phase.*

* **Preprocessing:** Cleaning, imputation (Median/Mode), and outlier treatment (Winsorizing).
* **Feature Engineering:** Encoding strategies (Target/One-Hot) and engineering User, Product, and Temporal interaction features.
* **Dimensionality & Collinearity:** Identified multicollinearity using VIF (Variance Inflation Factor) and applied regularization.
* **Imbalanced Data Handling:** Experimented with Class Weights and Sampling (SMOTE) for classification tasks.
