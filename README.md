# IBM-Watson-Churn-Analysisis
Analysis of customer churn using IBM Watson data set.
Telco Customer Churn Prediction & Retention Strategy
# Project Overview
This project focuses on building a predictive model to identify customers at high risk of churning from a telecommunications company. By understanding the key drivers of churn, the goal is to enable proactive, targeted interventions to improve customer retention, safeguard revenue, and enhance overall business profitability.

# Business Problem
Customer churn poses a significant financial threat to telecom companies. The cost of acquiring new customers far exceeds that of retaining existing ones, and each churned customer represents a direct loss of recurring revenue and market share. This project aimed to address the critical business questions:

Who is likely to churn? Identifying high-risk customers before they leave.

Why do customers churn? Uncovering the primary factors and characteristics driving attrition.

How can we intervene effectively? Developing data-driven, tailored retention strategies for different customer segments.

# Data Used
The analysis was performed using the IBM Watson Telco Customer Churn dataset.

# Source: Kaggle - Telecom Churn Dataset (IBM Watson Analytics)

Description: This dataset comprises 7,043 customer records with 21 features. Key columns include:

customerID: Unique identifier for each customer.

gender, SeniorCitizen, Partner, Dependents: Demographic information.

tenure: Number of months the customer has stayed with the company.

PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Information about services subscribed.

Contract: Contract term (Month-to-month, One year, Two year).

PaperlessBilling, PaymentMethod: Billing preferences.

MonthlyCharges, TotalCharges: Billing amounts.

Churn: The target variable, indicating whether the customer churned (Yes/No).

# Data Cleaning and Preparation
A robust data preprocessing pipeline was implemented to prepare the raw data for machine learning:

Missing Value Handling: Identified and removed 11 rows with missing values in the TotalCharges column initially non-numeric.

Target Variable Encoding: The Churn column was transformed from categorical ('Yes', 'No') to numerical (1, 0) for model compatibility.

Categorical Feature Encoding: All nominal categorical features (e.g., gender, Contract, InternetService, PaymentMethod) were converted into a numerical format using One-Hot Encoding (drop_first=True to prevent multicollinearity).

Numerical Feature Scaling: Continuous numerical features (tenure, MonthlyCharges, TotalCharges, SeniorCitizen) were scaled using StandardScaler. This ensures that features with larger numerical ranges do not disproportionately influence distance-based algorithms, improving model convergence and performance.

Feature Engineering: New, potentially more informative features were created to capture complex relationships:

MonthlyToTotalChargeRatio: Ratio of monthly charges to total charges.

NumServices: A count of the number of services a customer subscribes to.

Tenure_MonthlyCharges_Interaction: An interaction term to capture the combined effect of customer longevity and monthly spending.

# Machine Learning Models
The prepared dataset was split into training (75%) and testing (25%) sets, with stratification on the Churn variable to maintain class proportions due to the dataset's imbalance.

Several classification models were trained and evaluated:

Logistic Regression: Chosen as a strong, interpretable baseline model.

Decision Tree Classifier: Provides clear decision rules but can be prone to overfitting.

Random Forest Classifier: An ensemble method known for its robustness and good generalization.

Gradient Boosting Classifier: A powerful ensemble technique often achieving high predictive performance.

# Hyperparameter Tuning
Extensive hyperparameter tuning was performed using RandomizedSearchCV for both the Gradient Boosting Classifier and the Logistic Regression model. This process systematically searched for the optimal combination of hyperparameters (e.g., learning_rate, max_depth, C, penalty) to maximize the model's F1-score, a crucial metric for imbalanced datasets.

# Model Selection
After comprehensive evaluation across metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC, the Tuned Logistic Regression model was selected as the best performer. It achieved a high Recall of 0.797 and an F1-Score of 0.613, demonstrating its superior ability to correctly identify actual churners, which was the primary business objective. Its competitive ROC-AUC of 0.840 also highlighted its strong discriminatory power, coupled with its inherent interpretability. This interpretability is crucial for a Business Analyst, allowing clear communication of 'why' customers churn to business stakeholders.

# Key Findings & Business Recommendations
Translating these insights into actionable strategies, the model identified key churn drivers and informed the development of targeted retention campaigns:

Key Churn Drivers:
Contract Type (Month-to-month): The single strongest predictor of churn. Customers on shorter, month-to-month contracts are significantly more likely to churn.

Internet Service (Fiber Optic): Fiber optic users exhibit a high propensity to churn, suggesting potential dissatisfaction with service quality or intense competition in this segment.

Payment Method (Electronic Check): Customers using electronic checks are at a higher risk of churning, possibly due to billing clarity issues or overall dissatisfaction.

Tenure: Shorter tenure (newer customers) is a strong indicator of churn.

Lack of Add-on Services (Online Security, Tech Support): Customers without these protective and supportive services are more vulnerable to churn.

# Targeted Retention Strategies:
Three distinct churner segments were identified, each with tailored recommendations:

1. Segment: "The Untethered Fiber Optic Churners"
Characteristics: Month-to-month contracts, Fiber optic internet, Electronic check payment, Lower tenure, often lack security/support add-ons.

Strategy: Proactive Engagement & Value Bundle Conversion. Offer discounted 1-year contracts (10-15% off), free 3-month trials of Online Security/Tech Support, and a personalized billing clarity review within their first 3-6 months.

Estimated Impact: Projecting an additional 8% retention in this segment, leading to an estimated ~516% ROI resulting in an estimated annual revenue retention of ~$188,160 against an estimated cost of ~$30,536.

2. Segment: "The High-Cost, Unsecured Customers"
Characteristics: High monthly charges, few security-related add-on services, possibly senior citizens.

Strategy: Value Optimization & Personalized Education. Offer free personalized plan consultations, suggest more cost-effective bundles, and provide digital literacy support (e.g., workshops for seniors).

Estimated Impact: Projecting an additional 6% retention in this segment, leading to an estimated ~500% ROI resulting in an estimated annual revenue retention of ~$28,800 against an estimated cost of ~$4,800.

3. Segment: "The Short-Term, Basic Service Users"
Characteristics: Very low tenure (new customers), month-to-month contracts, often only basic phone service, may lack Tech Support/Device Protection.

Strategy: Enhanced Onboarding & Early Loyalty Building. Implement a "First 90 Days" program with proactive check-in calls, a "Welcome Bonus" upgrade (e.g., free month of a premium add-on), and early feedback surveys.

Estimated Impact: Projecting an additional 10% retention in this critical early period, leading to an estimated ~592% ROI by saving considerable annual revenue.

Tools Used
Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy (for RandomizedSearchCV distributions)

Development Environment: VS Code (Jupyter Notebook)

Version Control: Git & GitHub

# How to Run This Project
Clone the Repository:

git clone https://github.com/[Your GitHub Username]/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction


Set up Environment:

# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn


Open Jupyter Notebook:

jupyter notebook Telco_Churn_Analysis.ipynb


(Adjust Telco_Churn_Analysis.ipynb to your actual notebook filename).

Run Cells: Execute all cells in the Jupyter Notebook sequentially to reproduce the analysis, model training, and segmentation.
