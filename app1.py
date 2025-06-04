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

    This is a web application built using **Streamlit, a Python library**, that utilizes Machine Learning to detect whether a message is **Spam** or **Not Spam**.
    
    **Model Evaluation** :
    Accuracy  : 98.3% and
    Precision : 100%
    
    **Algorithm used** : Naive Bayes especially the Bernoulli Naive Bayes with TF-IDF Vectorization.
    
    **Project by :** Syed Yaseen (24466-CM-095) 
    Usha Rama College of Engineering & Technology,
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
    if not input_sms.strip():  # If input is empty or only spaces and empty string in python=False and by "not" it becomes True
        st.warning("Please enter a message before clicking Predict.")
    else:    
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
            st.markdown("#### Description:")
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
            
            spam_words = [('call', 302), ('free', 191), ('txt', 130), ('u', 119), ('ur', 119), ('mobile', 105), ('text', 104), ('stop', 104), ('claim', 96), 
                          ('reply', 96), ('prize', 81), ('get', 70), ('new', 64), ('send', 58), ('urgent', 57), ('nokia', 54), ('cash', 51), ('contact', 51), 
                          ('please', 49), ('service', 48), ('win', 47), ('c', 45), ('phone', 43), ('guaranteed', 42), ('per', 41), ('week', 40), ('customer', 40), 
                          ('tone', 38), ('chat', 36), ('cs', 35)]

            ham_words = [('u', 883), ('get', 293), ('gt', 288), ('lt', 287), ('go', 240), ('got', 236), ('know', 225), ('like', 221), ('ok', 217), ('good', 212), ('come', 211), ('ur', 197),
                         ('time', 188), ('call', 184), ('love', 172), ('day', 166), ('going', 164), ('want', 159), ('lor', 159), ('one', 158), ('home', 152), ('need', 151), ('still', 143),
                         ('da', 141), ('see', 135), ('back', 127), ('think', 126), ('today', 121), ('sorry', 121), ('n', 120)]


            with st.expander("**Top 30 Words used in Spam and Not Spam Messages**"):
                col1, col2 = st.columns(2)
            
                with col1:
                    st.markdown("#### Spam")
                    for word, count in spam_words:
                        st.write(f"**{word}** — {count} times")
            
                with col2:
                    st.markdown("#### Not Spam")
                    for word, count in ham_words:
                        st.write(f"**{word}** — {count} times")
            
                            
            
            
