import streamlit as st
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle

st.title('Spam Messages Classfier and Recommendation System')

X_train = pickle.load(open('X_train_data.pkl', 'rb'))
Y_train = pickle.load(open('Y_train_data.pkl', 'rb'))
sms_df= pickle.load(open('sms_df.pkl', 'rb'))


feature_extraction = CountVectorizer(stop_words='english')
X_train_features_bow = feature_extraction.fit_transform(X_train)
Y_train = Y_train.astype('int')

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features_bow, Y_train)

# Creating Streamlit app

image = Image.open('Spam_img2.png')
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image, caption='SPAM ALERT')
image = Image.open('Spam_img.png')
st.sidebar.image(image, width=120)
st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter a message:", "Congratulations! You've won a $1000 cash prize. Claim your prize now by clicking the link: http://example.com/claim", height=300)
# Making prediction on the user input
input_data_features = feature_extraction.transform([user_input])
prediction = model.predict(input_data_features)

def message(sms):
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.error("Spam SMS")
        st.subheader("Similar Spam Messages:")
        query_vec = feature_extraction.transform([user_input])
        word_similarity = cosine_similarity(query_vec, X_train_features_bow)[0]
        most_similar = sorted(list(enumerate(word_similarity)), reverse=True, key=lambda x: x[-1])[:5]
        res=[]
        for i in most_similar:
            res.append(st.write(sms_df.iloc[X_train.index[i[0]], 2]))
        return res
    else:
        return st.success("Non-Spam SMS")

if st.sidebar.button('Check Message'):
    print(message(user_input))
