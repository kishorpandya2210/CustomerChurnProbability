import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import utils as ut

load_dotenv()

client = OpenAI(
  base_url = "https://api.groq.com/openai/v1",
  api_key = os.getenv("GROQ_API_KEY")
)

def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model("../models/xgb_model.pkl")

naive_bayes_model = load_model("../models/nb_model.pkl")

random_forest_model = load_model("../models/rf_model.pkl")

decision_tree_model = load_model("../models/dt_model.pkl")

svm_model = load_model("../models/svm_model.pkl")

knn_model = load_model("../models/knn_model.pkl")

voting_classifier_model = load_model("../models/voting_clf.pkl")

xgboost_SMOTE_model = load_model("../models/xg-boost-SMOTE.pkl")

xgboost_featureEngineering_model = load_model(
  "../models/xgboost-featureEngineered.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):

  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany': 1 if location == 'Germany' else 0,
    'Geography_Spain': 1 if location == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0
  }

  input_df = pd.DataFrame({k: [v] for k, v in input_dict.items()})
  return input_df, input_dict

def make_predictions(input_df, input_dict):

  probabilites ={
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'RandomForest': random_forest_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilites.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Average probability of churn: {avg_probability:.2f}%")

  with col2:
    fig_probs = ut.create_prob_chart(probabilites)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown('### Model Probabilities')
  for model, prob in probabilites.items():
    st.write(f"{model} {prob}")
  st.write(f"Average Probability: {avg_probability}")
  return avg_probability


def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.
You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.
Here is the customer's information:
{input_dict}
Here is some explanation as to why the customer might be at risk
of churning:
{explanation}
Generate an email to the customer based on their information,
asking them to stay if they are at risk of churning, or offering them incentives so they become more loyal to the bank.
Make sure to list out a set of incentives to stay based on their
information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
"""

  raw_response = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[
          {"role": "user", "content": prompt}
      ],
  )
  print("\n\nEMAIL PROMPT", prompt)
  return raw_response.choices[0].message.content



def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are a highly skilled data scientist at a bank, specializing in interpreting and communicating the predictions of machine learning models.

- The machine learning model has predicted that the customer **{{surname}}** has a **{{round(probability * 100, 1)}}%** probability of churning.  
- Below is the customer's information:  

  {input_dict}

  ### Top 10 Most Important Features for Predicting Churn
| **Feature**          | **Importance** |
|-----------------------|----------------|
| NumOfProduct         | 0.323888       |
| IsActiveMember       | 0.164146       |
| Age                  | 0.109550       |
| Geography_Germany    | 0.091373       |
| Balance              | 0.052786       |
| Geography_France     | 0.046463       |
| Gender_Female        | 0.045283       |
| Geography_Spain      | 0.036855       |
| CreditScore          | 0.035005       |
| EstimatedSalary      | 0.032655       |
| HasCrCard            | 0.031940       |
| Tenure               | 0.030054       |
| Gender_Male          | 0.000000       |

  {pd.set_option('display.max_columns', None)}

  - Below are the **summary statistics** for:  
1. **Churned Customers**:  
  {df[df['Exited']==1].describe()}

2. **Non-Churned Customers**:
  Here are summary statistics for non-churned customers:
  {df[df['Exited']==0].describe()}
---

### Instructions:  
1. If the customer has a **risk of churning greater than 40%**, provide a **3-sentence explanation** highlighting the key reasons for their risk. Focus on the most impactful features.  
2. If the customer has a **risk of churning less than 40%**, or if the explanation is general, provide a concise explanation of the prediction. Avoid generic statements like “Based on the machine learning model’s prediction and top features.”  
3. Ensure your explanation is tailored to the customer’s specific data and aligns with the feature importances listed.  
"""
     
  print("EXPLAINATION PROMPT:", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[
        {"role": "user", "content": prompt}
    ]
  )
  return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")


df = pd.read_csv("../data/churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

  selected_customer_id = int(selected_customer_option.split(" - ")[0])

  print("Selected customer ID:", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]

  print("Surname", selected_surname)

  selected_customer = df.loc[df['CustomerId']==selected_customer_id].iloc[0]

  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input(
      "Credit Score",
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore'])
    )

    location = st.selectbox(
      "Location", ["Spain","France","Germany"],
      index = ["Spain","France","Germany"].index(selected_customer['Geography'])
    )
  
    gender = st.radio("Gender", ["Male","Female"],
                     index=0 if selected_customer['Gender'] == "Male" else 1)
  
    age = st.number_input(
      "Age",
      min_value=18,
      max_value=100,
      value = int(selected_customer['Age'])
    )

    tenure = st.number_input(
      "Tenure",
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure'])
    )

  with col2:
    balance = st.number_input(
      "Balance",
      min_value=0,
      value=int(selected_customer['Balance'])
    )

    num_products = st.number_input(
      "Number of Products",
      min_value=0,
      max_value=10,
      value=int(selected_customer['NumOfProducts'])
    )

    has_credit_card = st.checkbox(
      "Has Credit Card", 
      value=bool(selected_customer['HasCrCard'])
    )
    
    is_active_member = st.checkbox(
      "Is Active Member", 
      value=bool(selected_customer['IsActiveMember'])
    )

    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary'])
    )

                                   

  input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

  if st.button("Generate Churn Probability"):

    avg_probability = make_predictions(input_df, input_dict)
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

    st.session_state["avg_probability"] = avg_probability
    st.session_state["explanation"] = explanation
    st.session_state["email"] = email

  if "avg_probability" in st.session_state:
      if st.button("Generate Explanation and Email"):
        st.markdown("---")
        st.subheader("Explanation of Prediction")

        st.markdown(st.session_state["explanation"])
    
        st.markdown("---")
        st.subheader("Potential Email to Customer")

        st.markdown(st.session_state["email"])
