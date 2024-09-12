import requests
import json

# Sample input data
input_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1000,
    "LoanAmount": 2000,
    "Loan_Amount_Term": 12,
    "Credit_History": 1,
    "Property_Area": "Urban"
}

# Test all endpoints
endpoints = ['echo', 'logistic', 'random_forest', 'mlp', 'all']

for endpoint in endpoints:
    url = f'http://127.0.0.1:5000/predict/{endpoint}' if endpoint != 'echo' else 'http://127.0.0.1:5000/echo'
    response = requests.post(url, json=input_data)
    
    print(f"\nResponse from {endpoint.capitalize()} endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

# Additional test for the 'all' endpoint
if 'all' in endpoints:
    all_response = requests.post(f'http://127.0.0.1:5000/predict/all', json=input_data)
    if all_response.status_code == 200:
        result = all_response.json()
        print("\nBest Model Analysis:")
        print(f"Best Model: {result['best_model']}")
        print(f"Overall Recommendation: {result['overall_recommendation']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Risk Assessment: {result['risk_assessment']}")