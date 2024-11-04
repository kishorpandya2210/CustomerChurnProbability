import requests
import json

url = "https://customerchurnprobability.onrender.com/predict"

customer_data={
    "CreditScore": 700,
    "Age": 20,
    "Tenure": 0,
    "Balance": 0,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 0,
    "Gender": "Male",
    "Geography": "Germany",
}

response = requests.post(url, json=customer_data)

if response.status_code==200:
    result = response.json()
    print(result)
else:
    print("Error", response.status_code, response.text)