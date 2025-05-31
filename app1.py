import sys
sys.path.append("/ur project/myenvi/lib/site-packages")
import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Text preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Sidebar block
st.sidebar.title("About")
st.sidebar.info(
    """
    **Email/SMS Spam Classifier**

    This is a web application built using Machine learning technology that detects whether a message is **Spam** or **Not Spam**.
    
    **Model Evaluation** :
    Accuracy  : 98.3%
    Precision : 100%
    
    **Algorithm used** : Naive Bayes especially the Bernoulli Naive Bayes with TF-IDF Vectorization.
    
    **Project by :** Syed Yaseen (24466-CM-095) 
    Usha Rama College of Engineering & Technology
    Telaprolu.
    """
)

st.sidebar.markdown("[View Project Report (PDF)](https://drive.google.com/file/d/1s8rVXgzoeY24yjGvWcIto28GpfnUkZhE/view?usp=sharing)")

# App Title
st.title("Email/SMS Spam Classifier")

# Sample Messages block
example_sms = st.selectbox(
    "Choose a sample message (optional):",
    ["", 
     "congratulations! you have won 1000 call on this number to get your prize", 
     "Hey, are you coming to class today?", 
     "You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.", 
     "Meet me at the cafe at 5.",
     "I am free today, lets go out for a movie. What do you say?",
     "A [redacted] loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply",
     "Did you see the match? It was insane"]
)

# Use selected sample message as default input
input_sms = st.text_area("Enter your message below:", example_sms)

# Predict Button + Spinner block
if st.button("Predict"):
    with st.spinner("Analyzing message..."):
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Probabilities as percentages
        prob_not_spam = round(proba[0] * 100, 2)
        prob_spam = round(proba[1] * 100, 2)

    
        # Header + Explanation block
    

        # Header
        if result == 1:
            st.header("**Spam**")
        else:
            st.header("**Not Spam**")

        # Explanation section
        st.markdown("---")
        st.subheader("Description:")
        st.write(f"Probability of Spam class: **{prob_spam}%**")
        st.write(f"Probability of Not Spam class: **{prob_not_spam}%**")

        if result == 1:
            st.markdown(
                "Since the probability of **Spam** class is greater than **Not Spam** class, "
                "it was classified as **Spam**."
            )
        else:
            st.markdown(
                "Since the probability of **Not Spam** class is greater than **Spam** class, "
                "it was classified as **Not Spam**."
            )
