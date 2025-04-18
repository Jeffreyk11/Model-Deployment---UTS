import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('xg_loan.pkl')
scaler = joblib.load('scaler.pkl')
gender_encode = joblib.load('label_encoder_gender.pkl')
prev_encode = joblib.load('label_encoder_prev.pkl')
edu_encode = joblib.load('ordinal_encoder_education.pkl')
home_encode = joblib.load('ordinal_encoder_home.pkl')
ohe_encode = joblib.load('ohe_encoder_loan_intent.pkl')



def main():
    st.title('📊 Loan Status Prediction App')
    st.write("Masukkan data nasabah untuk memprediksi status pinjaman:")

    person_age = st.number_input('Age', 18, 100)
    person_gender = st.selectbox('Gender', ['Male', 'Female'])
    person_education = st.selectbox('Education Level', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input('Monthly Income', 0,10000000000000)
    person_emp_exp = st.number_input('Employment Experience (years)', 0, 100)
    person_home_ownership = st.selectbox('Home Ownership', ['OTHER', 'RENT', 'MORTGAGE', 'OWN'])
    loan_amnt = st.number_input('Loan Amount', 0,500000)
    loan_intent = st.selectbox('Loan Purpose', ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_int_rate = st.number_input('Interest Rate (%)', 0.0, 100.0,)
    loan_percent_income = st.number_input('Loan % of Income', 0.0, 1.0, 0.0)
    cb_person_cred_hist_length = st.number_input('Credit History Length (years)', 0, 100, 0)
    credit_score = st.number_input('Credit Score', 0, 1000, )
    previous_loan_defaults_on_file = st.selectbox('Has Previous Defaults?', ['Yes', 'No'])
    
    raw_data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }

    df = pd.DataFrame([raw_data])
    df['person_gender'] = df['person_gender'].str.lower()
    
    df['person_gender'] = gender_encode.transform(df['person_gender'])
    df['previous_loan_defaults_on_file'] = prev_encode.transform(df['previous_loan_defaults_on_file'])

    df['person_education'] = edu_encode.transform(df[['person_education']])
    df['person_home_ownership'] = home_encode.transform(df[['person_home_ownership']])

    ohe_transformed = ohe_encode.transform(df[['loan_intent']])
    ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_encode.get_feature_names_out(['loan_intent']))
    df = pd.concat([df.drop(columns=['loan_intent']), ohe_df], axis=1)

    numerical_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    if st.button('🔍 Predict Loan Status'):
        try:
            prediction = model.predict(df)
            result = "Yes" if prediction[0] == 1 else "No"
            st.success(f'✅ Predicted Loan Status: **{result}**')
        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")


if __name__ == '__main__':
    main()