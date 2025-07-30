import streamlit as st
import pandas as pd
import joblib

# Load the trained model and model columns
try:
    model = joblib.load('best_rf_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model or column file not found. Please ensure 'best_rf_model.pkl' and 'model_columns.pkl' are in the same directory.")
    st.stop()

def predict_rating(movie_features):
    """
    Predicts the IMDB rating for a given movie using the loaded model.
    """
    # Create a DataFrame from the input features
    input_df = pd.DataFrame([movie_features])

    # One-hot encode the categorical features
    # The 'columns' parameter ensures that if a genre/director was not in the training data, it is handled correctly.
    input_df = pd.get_dummies(input_df, columns=['Genre', 'Director'])

    # Align the columns of the input DataFrame with the columns of the training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict the rating
    predicted_rating = model.predict(input_df)

    return predicted_rating[0]

# --- Streamlit App Interface ---

st.title('🎬 IMDB Movie Rating Predictor')

st.write("Enter the details of a movie to predict its IMDB rating.")

# Create input fields for the user
released_year = st.number_input('Released Year', min_value=1920, max_value=2025, value=2022, step=1)
genre = st.text_input('Genre', 'Action, Crime, Drama')
director = st.text_input('Director', 'Christopher Nolan')
runtime = st.number_input('Runtime (in minutes)', min_value=30, max_value=300, value=150, step=1)

# Create a button to trigger the prediction
if st.button('Predict Rating'):
    movie_to_predict = {
        'Released_Year': released_year,
        'Genre': genre,
        'Director': director,
        'Runtime': runtime
    }

    try:
        prediction = predict_rating(movie_to_predict)
        st.success(f"The predicted IMDB rating is: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
