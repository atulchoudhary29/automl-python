### Python 3.7 - Python 3.10 required due to pycaret

import os
import pickle
import numpy as np
import pandas as pd
import pandas_profiling
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from streamlit_pandas_profiling import st_profile_report
import warnings
warnings.filterwarnings("ignore")

# --- FUNCTIONS ---
def preprocess_data(df):
    """
    Preprocesses the input dataframe.
    - Converts object columns to string
    - Converts boolean columns to integer (1 for True, 0 for False)
    - Label encodes categorical columns
    """
    # Convert object columns to string
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype(str)
    
    # Convert boolean columns to integer
    for col in df.select_dtypes(['bool']).columns:
        df[col] = df[col].astype(int)
    
    # Label encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders


# Home section information and details
def home_info():
    st.title("Welcome to AutoML: Automated Machine Learning")
    st.write("""
    Machine learning (ML) is a subset of artificial intelligence that involves 
    training algorithms to recognize patterns and make decisions based on data. 
    These algorithms can be used for a wide range of tasks, from image recognition 
    to predicting stock prices.
    
    In ML, there are primarily two types of tasks:
    
    1. **Regression**: Predicting a continuous value. For example, predicting the price of a house based on its features.
    
    2. **Classification**: Predicting a category or class. For example, determining if an email is spam or not spam.
    
    Choosing between regression and classification mainly depends on the nature of your target variable.
    
    **Instructions for this App**:
    1. **Upload**: Start by uploading your dataset. This will be the foundation for all other tasks.
    2. **Data Cleaning**: Clean the data by handling missing values.
    3. **Profiling**: Get insights about your dataset with an exploratory data analysis report.
    4. **Modelling**: Train a machine learning model on your data.
    5. **Test Model**: Test your trained model with new data to see its predictions.
    6. **Download**: Download your trained model for use elsewhere.
    
    Navigate through each section using the sidebar, and enjoy your data exploration journey!
    """)

# --- MAIN CODE ---
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv')

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML: Automated Machine Learning")
    st.write(
        "Welcome to the AutoML app! Simplify your machine learning journey, whether you're a data scientist, business analyst, or just curious. No deep programming needed—just follow the navigation!")
    # Adding the Home choice in the Navigation
    choice = st.radio("Navigation", ["Home", "Upload", "Data Cleaning", "Profiling", "Modelling", "Test Model", "Download"])

# Check if dataset is in global memory
dataset_exists = 'df' in globals()

if choice == "Home":
    home_info()

elif choice == "Upload":
    st.title("Upload Your Dataset")
    st.write(
        "Start by uploading your dataset here. This data will be used to train a machine learning model. "
        "Please ensure it's a CSV (Comma Separated Values) file."
    )
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

elif choice == "Data Cleaning":
    st.title("Data Cleaning Options")
    st.write(
        "Before training a model, it's essential to clean the dataset. In this section, "
        "you can handle missing values by removing the rows containing them. "
        "Press the checkbox to clean the data."
    )
    if dataset_exists:
        # Handle missing values
        if st.checkbox('Handle Missing Values'):
            df.dropna(inplace=True)
            st.dataframe(df)
    else:
        st.warning("Dataset not uploaded.")

elif choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    st.write(
        "Get insights about your dataset! This section provides a comprehensive report "
        "that showcases various properties and characteristics of your data."
    )
    if dataset_exists:
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Dataset not uploaded.")

elif choice == "Modelling": 
    st.title("Model Training")
    model_type = st.radio("Model Type", ["Classification", "Regression"])
    st.write(
        "Here, the magic happens! Choose the columns you want to use for prediction "
        "and the target column you want to predict. The app will then automatically "
        "select and train the best model for you. You can also tune the model for better accuracy."
    )

    if dataset_exists:
        # Model Training Process
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        chosen_inputs = st.multiselect('Choose Input Columns', df.columns, default=df.columns.drop(chosen_target).tolist())
        with open('selected_columns.txt', 'w') as f:
            for column in chosen_inputs:
                f.write(f"{column}\n")
        
        if chosen_target in chosen_inputs:
            st.warning("Target column should not be in input columns. Please deselect it.")

        tune_model = st.checkbox("Tune Model for better accuracy?")
        if st.button('Run Modelling'):
            try:
                # Using only the selected columns
                df = df[chosen_inputs + [chosen_target]]
                df, label_encoders = preprocess_data(df)
                
                if model_type == "Classification":
                    from pycaret.classification import compare_models, pull, save_model, setup
                    automl_env = setup(df, target=chosen_target, verbose=False)
                    best_model = compare_models(n_select=(3 if tune_model else 1))
                
                elif model_type == "Regression":
                    from pycaret.regression import compare_models, pull, save_model, setup
                    automl_env = setup(df, target=chosen_target, verbose=False)
                    best_model = compare_models(n_select=(3 if tune_model else 1))
                with open('model_type.txt', 'w') as f:
                    f.write(model_type)
                st.info("Machine Learning Model Comparison")
                st.dataframe(pull())
                
                if isinstance(best_model, list):
                    save_model(best_model[0], 'best_model')
                else:
                    save_model(best_model, 'best_model')
                with open('label_encoders.pkl', 'wb') as f:
                    pickle.dump(label_encoders, f)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # Evaluation metrics explanation
            if model_type == "Classification":
                st.subheader("Explanation of Classification Metrics:")
                st.markdown("""
                - **Accuracy**: Proportion of all predictions that are correct.
                - **Precision**: Proportion of positive identifications that were actually correct.
                - **Recall (Sensitivity)**: Proportion of actual positives that were identified correctly.
                - **F1-Score**: Harmonic mean of Precision and Recall, ranges between 0 (worst) and 1 (best).
                - **AUC**: Area Under the Receiver Operating Characteristic Curve, measures the ability of the model to distinguish between the positive and negative classes.
                """)
            elif model_type == "Regression":
                st.subheader("Explanation of Regression Metrics:")
                st.markdown("""
                - **Mean Absolute Error (MAE)**: Average of the absolute differences between predicted and actual values.
                - **Mean Squared Error (MSE)**: Average of the squared differences between predicted and actual values.
                - **Root Mean Squared Error (RMSE)**: Square root of the MSE, provides the error magnitude in the same units as the predicted value.
                - **R-squared**: Proportion of the variance in the dependent variable that is predictable from the independent variables, ranges between 0 (worst) and 1 (best).
                """)

    else:
        st.warning("Dataset not uploaded.")

elif choice == "Test Model":
    st.title("Test Your Model")
    st.write(
        "Once you have a trained model, you can input new data here to see its predictions. "
        "Enter the data values separated by commas. For example, if your dataset has columns A, B, and C, "
        "you might enter something like '5, Yes, Blue'."
    )
    if os.path.exists('best_model.pkl'):
        if os.path.exists('model_type.txt'):
            with open('model_type.txt', 'r') as f:
                model_type = f.read().strip()
        if model_type == "Classification":
            from pycaret.classification import load_model, predict_model
            model = load_model('best_model')
        elif model_type == "Regression":
            from pycaret.regression import load_model, predict_model
            model = load_model('best_model')

        if os.path.exists('selected_columns.txt'):
            with open('selected_columns.txt', 'r') as f:
                selected_columns = [line.strip() for line in f.readlines()]
            st.write(f"Original Columns (excluding target): {selected_columns}")
            input_data = st.text_input(f"Enter your data for prediction for columns {selected_columns} (comma-separated values):")
        
        if st.button("Predict"):
            try:
                with open('label_encoders.pkl', 'rb') as f:
                    label_encoders = pickle.load(f)
                
                # Data processing for prediction
                input_list = input_data.split(',')
                input_list = [1 if i.strip() == 'True' else (0 if i.strip() == 'False' else i) for i in input_list]
                test_data = pd.DataFrame([input_list], columns=selected_columns)
                
                for col in selected_columns:
                    if df[col].dtype != 'object':
                        test_data[col] = test_data[col].astype(float)
                
                st.write("Inputted Data:")
                st.text(test_data.to_string(index=False))
                
                for col, le in label_encoders.items():
                    if col in test_data:
                        test_data[col] = test_data[col].map(lambda x: x if x in le.classes_ else 'Unknown')
                        le.classes_ = np.append(le.classes_, 'Unknown')
                        test_data[col] = le.transform(test_data[col])
                
                prediction = predict_model(model, data=test_data)
                st.write(f"Predicted Label: {prediction['prediction_label'].iloc[0]}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif choice == "Download": 
    st.title("Download Your Trained Model")
    st.write(
        "You can download the model you've trained and use it elsewhere. "
        "Click the download button to get your model file."
    )
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model found to download.")
