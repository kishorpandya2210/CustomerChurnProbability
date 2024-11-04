from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the model
with open("./models/xgb_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Preprocess data function
def preprocess_data(customer_dict):
    input_dict = {
        "CreditScore": customer_dict["CreditScore"],
        "Age": customer_dict["Age"],
        "Tenure": customer_dict["Tenure"],
        "Balance": customer_dict["Balance"],
        "NumOfProducts": customer_dict["NumOfProducts"],
        "HasCrCard": customer_dict["HasCrCard"],
        "IsActiveMember": customer_dict["IsActiveMember"],
        "EstimatedSalary": customer_dict["EstimatedSalary"],
        "Geography_France": 1 if customer_dict["Geography"] == "France" else 0,
        "Geography_Germany": 1 if customer_dict["Geography"] == "Germany" else 0,
        "Geography_Spain": 1 if customer_dict["Geography"] == "Spain" else 0,
        "Gender_Male": 1 if customer_dict["Gender"] == "Male" else 0,
        "Gender_Female": 1 if customer_dict["Gender"] == "Female" else 0
    }

    customer_df = pd.DataFrame([input_dict])
    return customer_df

# Get predictions function
def get_predictions(customer_dict):
    preprocessed_data = preprocess_data(customer_dict)  # Renamed variable to preprocessed_data
    predictions = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)

    return predictions, probability

@app.post("/predict")
async def predict(data: dict):
    predictions, probability = get_predictions(data)
    return {
        "predictions": predictions.tolist(),  # Convert to list for JSON serialization
        "probability": probability.tolist()   # Convert to list for JSON serialization
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
