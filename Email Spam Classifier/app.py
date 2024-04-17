import base64
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

#Loading the saved vectorizer and Naive Bayes Model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# transform)text function for textt preprocessing
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('yellow.jpg')

plt.style.use("ggplot")


def transform_text(text):
    text = text.lower()  # Converting to lowercase
    text = nltk.word_tokenize(text)   # Tokenize

    # Removing special character and retaining aplhanumeric words
    text = [word for word in text if word.isalnum()]

    # Removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Streamlit Code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")


if st.button('Predict'):

    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
