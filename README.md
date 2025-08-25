
Loan Approval Prediction:

📌 Objective:
Predict whether a loan application will be approved (1) or rejected (0) based on applicant and credit attributes.

📊 Dataset:

Categorical Features: Gender, Married, Education, Self_Employed, Property_Area

Numerical Features: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

Target: Loan_Status (Y/N → 1/0)

Dropped: Loan_ID (identifier)

⚙️ Steps Performed

Data Cleaning & Preprocessing:

Removed duplicates, handled missing values (median for numeric, mode for categorical).

Encoded categorical variables with One-Hot Encoding.

Fixed right-skewed numeric features with log transformation.

Exploration & Feature Importance:

Correlation analysis showed Credit_History as the most influential factor.

Top features: ApplicantIncome, Credit_History, CoapplicantIncome.

Modeling:

Train/Test split (80/20), SMOTE applied for class balancing.

RandomForestClassifier trained with scaled features.

Evaluation:

Accuracy: ~95%

Confusion Matrix showed good balance between approvals and rejections.

Both classes achieved high precision and recall.

🛠️ Tech Stack:

Python: pandas, numpy, matplotlib, seaborn

Modeling: scikit-learn, imblearn (SMOTE)

Encoding: category_encoders

✅ Results:

Loan approval prediction achieved ~95% accuracy.

Credit_History was the most significant predictor.
