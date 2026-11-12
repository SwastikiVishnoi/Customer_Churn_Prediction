import streamlit as st
import pandas as pd
import joblib

# Load the files saved from the Colab notebook
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')
defaults = joblib.load('defaults.pkl')

st.title('Customer Churn Predictor')
st.write('Enter a customer\'s details to estimate the chance that they will leave.')

# Inputs the user can change
tenure = st.slider('Tenure in months', 0, 72, 12)
monthly = st.slider('Monthly charge ($)', 18, 120, 70)
referrals = st.slider('Number of referrals', 0, 11, 0)
contract = st.selectbox('Contract', ['Month-to-Month', 'One Year', 'Two Year'])
internet = st.selectbox('Internet type', ['Cable', 'DSL', 'Fiber Optic', 'None'])
payment = st.selectbox('Payment method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])
dependents = st.checkbox('Has dependents')
paperless = st.checkbox('Paperless billing')

threshold = st.slider('Flag as at-risk above this probability', 0.1, 0.9, 0.4)

if st.button('Predict'):
    # Start from a typical customer, then clear the groups we are setting
    row = defaults.copy()
    for col in columns:
        if col.startswith(('Contract_', 'Internet Type_', 'Payment Method_')):
            row[col] = 0.0

    # Numeric inputs
    row['Tenure in Months'] = tenure
    row['Monthly Charge'] = monthly
    row['Number of Referrals'] = referrals

    # Category inputs (the first option of each group was dropped during training)
    for name in [f'Contract_{contract}', f'Internet Type_{internet}', f'Payment Method_{payment}']:
        if name in columns:
            row[name] = 1.0
    if 'Dependents_Yes' in columns:
        row['Dependents_Yes'] = 1.0 if dependents else 0.0
    if 'Paperless Billing_Yes' in columns:
        row['Paperless Billing_Yes'] = 1.0 if paperless else 0.0

    # Same preparation as in training: same column order, then scale
    X_new = pd.DataFrame([row])[columns].astype(float)
    X_scaled = scaler.transform(X_new)
    probability = model.predict_proba(X_scaled)[0, 1]

    st.metric('Churn probability', f'{probability:.0%}')
    if probability >= threshold:
        st.error('At risk: consider a retention offer.')
    else:
        st.success('Low risk.')

    st.caption('Demo only. All other details are set to typical values.')
