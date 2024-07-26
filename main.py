import streamlit as st
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the TF-IDF vectorizer and logistic regression model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    custom_stopwords = set(STOPWORDS) - {'not'}
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower().split()  # Lowercase and split
    text = [stemmer.stem(word) for word in text if word not in custom_stopwords]  # Stemming and remove stopwords
    return ' '.join(text)


# Streamlit app
st.title("Sentiment Analysis of Reviews")
st.write("Enter a review text or upload a file to predict sentiment.")

# Text input
user_input = st.text_area("Enter a sentence for sentiment analysis:")

if st.button("Predict Sentiment (Text)"):
    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)

        # Transform the input text using the TF-IDF vectorizer
        input_tfidf = tfidf.transform([processed_input])

        # Scale the input data
        input_scaled = scaler.transform(input_tfidf.toarray())

        # Predict the sentiment
        prediction = model.predict(input_scaled)[0]

        # Display the result
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        st.write(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.write("Please enter a review text.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file containing column named 'Sentence'", type="csv")

if uploaded_file is not None:
    # Read the file into a DataFrame
    df = pd.read_csv(uploaded_file)

    if 'Sentence' not in df.columns:
        st.write("The uploaded file must contain a column named 'Sentence'.")
    else:
        # Preprocess and predict
        df['processed_text'] = df['Sentence'].apply(preprocess_text)
        df_tfidf = tfidf.transform(df['processed_text'])
        df_scaled = scaler.transform(df_tfidf.toarray())
        df['prediction'] = model.predict(df_scaled)
        df['sentiment'] = df['prediction'].apply(lambda x: 'POSITIVE' if x == 1 else 'NEGATIVE')

        # Save the results to a new CSV file
        output_file = 'predicted_sentiments.csv'
        df.to_csv(output_file, index=False)

        # Provide download link
        st.write("Predictions have been saved to 'predicted_sentiments.csv'.")
        st.download_button(label="Download Predictions", data=open(output_file, 'rb').read(), file_name=output_file,
                           mime='text/csv')
