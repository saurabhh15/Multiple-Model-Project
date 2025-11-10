import streamlit as st
import time
import pandas as pd
import random
import numpy as np
import pickle
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder


st.title('Hybrid Machine learning Project')
st.header('Select Dataset to predict value!!')
st.subheader('Project Summary:')

summary='''
This project predicts outcomes using five datasets—Car, Order, Student, Cancer, and Diabetes. Users can select features to generate predictions for both classification and regression tasks:

~ Car: Predicts car prices or categories based on specifications.

~ Order: Analyzes purchase patterns to predict order value or customer behavior.

~ Student: Predicts student performance or grades based on academic and personal factors.

~ Cancer: Classifies tumors as benign or malignant based on medical attributes.

~ Diabetes: Predicts diabetes progression or presence based on health indicators.

The system applies multiple machine learning models to provide insights tailored to each dataset.'''

st.write(summary)

st.sidebar.title('Select Project 🎯 ')

user_project_selection = st.sidebar.radio('Project List: ',['Car','Student','Order','Diabetes','Cancer'])

temp_df = pd.read_csv(user_project_selection.lower())
st.write(temp_df.sample(2))

np.random.seed(23)

X_all_input = []
for col in temp_df.columns[:-1]:  # exclude target column
    col_dtype = temp_df[col].dtype
    if col_dtype == object:
        options = temp_df[col].unique()
        choice = st.sidebar.selectbox(f'Select {col} value', options)
        X_all_input.append(choice)
        st.sidebar.write(f"You selected: {choice}")
    elif col_dtype == bool:
        options = [True, False]
        choice = st.sidebar.selectbox(f'Select {col} value', options)
        X_all_input.append(choice)
        st.sidebar.write(f"You selected: {choice}")
    else:  # numeric
        min_f, max_f = temp_df[col].agg(['min', 'max']).values
        if min_f == max_f:  # prevent slider crash
            max_f = max_f + 1
        choice = st.sidebar.slider(f'{col}', float(min_f), float(max_f),
                                   float(temp_df[col].sample(1).values[0]))
        X_all_input.append(choice)

X_input = pd.DataFrame([X_all_input], columns=temp_df.columns[:-1])
st.subheader('User Selected Choice:')
st.write(X_input)

# ------------------------------------------------------------
# Load model pipeline
# ------------------------------------------------------------
model_name = user_project_selection.lower()
final_model_name = model_name + '_ml_brain.pkl'

with open(final_model_name, 'rb') as f:
    chatgpt_brain = pickle.load(f)

# ------------------------------------------------------------
# ✅ FIX: Load encoded columns dictionary
# ------------------------------------------------------------
encoded_dict_name = 'encoded_columns_dict.pkl'
try:
    with open(encoded_dict_name, 'rb') as f:
        encoded_columns_dict = pickle.load(f)
except FileNotFoundError:
    st.warning("Encoded columns dictionary file not found. Using fallback.")
    encoded_columns_dict = {}

# ------------------------------------------------------------
# Handle categorical encoding alignment
# ------------------------------------------------------------
if user_project_selection.lower() in ['car', 'order', 'restraunt']:
    # Get encoded columns for current project
    encoded_cols = encoded_columns_dict.get(user_project_selection.lower() + '_df')
    if encoded_cols is None:
        st.error("Encoded columns for this project not found in dictionary.")
        st.stop()

    # Encode user input
    input_encoded = pd.get_dummies(X_input, drop_first=True, dtype=int)
    # Reindex to match training columns
    input_encoded = input_encoded.reindex(columns=encoded_cols, fill_value=0)
    final_X = input_encoded
else:
    final_X = X_input

# ------------------------------------------------------------
# Make prediction
# ------------------------------------------------------------
predicted_value = chatgpt_brain.predict(final_X)
final_predicted_value = predicted_value[0]

# ------------------------------------------------------------
# Display target-specific labels
# ------------------------------------------------------------
car_target_names = ['Car price']
student_target_names = ['Salary']
cancer_target_names = ['malignant', 'benign']
order_target_names = ['Price']
diabetes_target_names = ['diabetes progress']

target = None
ans_name = ''

if user_project_selection.lower() == 'student':
    target = student_target_names
    ans_name = 'Prediction is: '
elif user_project_selection.lower() == 'cancer':
    target = cancer_target_names
    ans_name = 'Prediction: '

# ------------------------------------------------------------
# Display final prediction
# ------------------------------------------------------------
st.subheader(ans_name)
st.write(final_predicted_value)
