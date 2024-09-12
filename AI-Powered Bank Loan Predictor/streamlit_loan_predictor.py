import streamlit as st
import requests
import json

# Set the API endpoint URLs
ALL_MODELS_URL = "http://localhost:5000/predict/all"
ECHO_URL = "http://localhost:5000/echo"

def main():
    st.title("Bank Loan Prediction")

    # Gather user input
    st.subheader("Enter Applicant Details")

    # Gender
    gender = st.radio("Gender", ['Male', 'Female'])

    # Marital Status
    married = st.selectbox("Marital Status", ['Unmarried', 'Married'])

    # Dependents
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])

    # Education
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])

    # Self-Employed
    self_employed = st.selectbox("Self-Employed", ['Yes', 'No'])

    # Applicant Income
    applicant_income = st.number_input("Applicant Income", min_value=0.0, step=1000.0)

    # Co-Applicant Income
    coapplicant_income = st.number_input("Co-Applicant Income", min_value=0.0, step=1000.0)

    # Loan Amount
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=10000.0)

    # Loan Amount Term
    loan_term = st.selectbox("Loan Amount Term (in months)", [12, 36, 60, 84, 120, 180])

    # Credit History
    credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", [1, 0])

    # Property Area
    property_area = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])

    # Make predictions
    if st.button("Predict Loan Approval"):
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }

        # Call the Flask API endpoint for all models
        response = requests.post(ALL_MODELS_URL, json=input_data)

        # Display the results
        if response.status_code == 200:
            result = response.json()
            
            st.subheader("Model Predictions")
            for prediction in result['individual_predictions']:
                with st.expander(f"{prediction['model']} Model", expanded=True):
                    st.json({
                        "approval_probability": round(prediction['approval_probability'], 2),
                        "loan_approved": prediction['loan_approved'],
                        "model": prediction['model'],
                        "risk_assessment": prediction['risk_assessment']
                    })
            
            st.subheader("Overall Recommendation")
            st.write(f"Best Model: {result['best_model']}")
            st.write(f"Recommendation: {result['overall_recommendation']}")
            st.write(f"Confidence: {result['confidence']:.2f}")
            st.write(f"Risk Assessment: {result['risk_assessment']}")
        else:
            st.error("Error making predictions. Please check the server logs.")

if __name__ == '__main__':
    main()
