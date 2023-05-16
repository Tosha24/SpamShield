import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer 

def transform_text(text):
    # 1. Lower case
    text = text.lower()
    
    # Tokenization - converting each word into list's element
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing Stop words and punctuations
    text = y.copy()
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    # Applying Stemming
    text = y.copy()
    y.clear()
    
    ps = PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))
                  
    return " ".join(y)

try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except:
    st.error('Model Loading error. Please check the model file path')
    
with open('image_url.txt', 'r') as f:
    image = f.read()

# Apply custom CSS styling to center the image
st.markdown(
    f"""
    <style>
    .centered-image {{
        position: absolute;
        bottom: -100px;
        left: -200px;
        display: flex;
    }}
    .centered-image img{{
    border-radius: 50%;
    contain: fit;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display the centered image
st.markdown(f'<div class="centered-image"><img src="{image}" alt="Centered Image"></div>', unsafe_allow_html=True)

st.title('SpamShield')
st.subheader('It is a SMS-Spam Classification System')

input_sms = st.text_area(label='Enter message to classify')

if st.button('Classify'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if input_sms == '':
        st.header("Please enter a SMS to classify")
    elif result == 1:
        st.header("It is a Spam SMS")
    else:
        st.header("It is not a Spam SMS")
