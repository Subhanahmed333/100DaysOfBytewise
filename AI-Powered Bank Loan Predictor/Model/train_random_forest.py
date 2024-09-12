import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine the datasets
data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Drop unnecessary columns
data = data.drop(['Loan_ID'], axis=1)

# Handle missing values
numeric_columns = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# Impute numeric columns with median
numeric_imputer = SimpleImputer(strategy='median')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Impute categorical columns with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Encode categorical variables
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le

# Prepare features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Encode target variable
le_loan_status = LabelEncoder()
y = le_loan_status.fit_transform(y.astype(str))
le_dict['Loan_Status'] = le_loan_status

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Print model performance
print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the Random Forest model
joblib.dump(rf_model, './Model/Random_Forest_Model.joblib')
print("Random Forest Model saved as Random_Forest_Model.joblib")

# Save the scaler (if not already saved)
joblib.dump(scaler, './Model/scaler.joblib')
print("Scaler saved as scaler.joblib")

# Save the label encoders (if not already saved)
joblib.dump(le_dict, './Model/label_encoders.joblib')
print("Label encoders saved as label_encoders.joblib")

# Save the column names (if not already saved)
import json
with open('./Model/column_names.json', 'w') as f:
    json.dump(list(X.columns), f)
print("Column names saved as column_names.json")