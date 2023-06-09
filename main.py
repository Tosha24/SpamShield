import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
    
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
    with open('vectorizer.pkl', 'rb') as file1, open('model.pkl', 'rb') as file2:
        try:
            tfidf = pickle.load(file1)
            model = pickle.load(file2)

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
        except Exception as e:
            st.header(e)

        finally:
            file1.close()
            file2.close()
