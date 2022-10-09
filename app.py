

import warnings

warnings.filterwarnings("ignore")
# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import flask
from flask import Flask, jsonify, request, render_template
from waitress import serve
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.layers import LSTM
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import nltk
nltk.download('vader_lexicon')
import nltk
nltk.download('stopwords')


# In[94]:
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

nltk.download('stopwords')


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def decontractions(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase


stopwords = stopwords.words('english')
def preprocess(text_col,stopword):
    preprocessed = []
    for sentence in (text_col.values):
        # Replace "carriage return" with "space".
        sentence=str(sentence)
        sent = sentence.replace('\\r', ' ')
        # Replace "quotes" with "space".
        sent = sent.replace('\\"', ' ')
        # Replace "line feed" with "space".
        sent = sent.replace('\\n', ' ')
        # Replace characters between words with "space".
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        #remove stop words
        #decontraction
        sent=decontractions(sent)
        #Remove emoji
        sent=deEmojify(sent)
        if stopword:
            sent = ' '.join(e for e in sent.split() if e not in stopwords)
        else:
           sent = ' '.join(e for e in sent.split())
        # to lowercase
        preprocessed.append(sent.lower().strip())
    return preprocessed


# In[101]:


import nltk
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def generate_sentiment_scores(data):
    sid = SentimentIntensityAnalyzer()
    neg=[]
    pos=[]
    neu=[]
    comp=[]
    for sentence in (data['parent_comment'].values): 
        sentence_sentiment_score = sid.polarity_scores(sentence)
        comp.append(sentence_sentiment_score['compound'])
        neg.append(sentence_sentiment_score['neg'])
        pos.append(sentence_sentiment_score['pos'])
        neu.append(sentence_sentiment_score['neu'])
    return comp,neg,pos,neu

@app.route("/predict", methods=['POST'])
def predict():
    Data = request.form.to_dict()
    data= pd.DataFrame([Data.values()], columns= Data.keys())

    with open('XGB.pkl', 'rb') as f:
        xgb_model= pickle.load(f)
    with open('tf1.pkl', 'rb') as f:
        tf1= pickle.load(f)
    with open('tf2.pkl', 'rb') as f:
        tf2= pickle.load(f)
    with open('tf3.pkl', 'rb') as f:
        tf3= pickle.load(f)
    with open('tf4.pkl', 'rb') as f:
        tf4= pickle.load(f)

    ups=data['ups'].astype(np.float64)
    score=data['score'].astype(np.float64)
    downs=data['downs'].astype(np.float64)
    data['comment']=preprocess(data['comment'],stopword=False)
    data['parent_comment']=preprocess(data['parent_comment'],stopword=False)
    data['compound'],data['negative'],data['positive'],data['neutral']=generate_sentiment_scores(data)
    data['com_len']=data['comment'].apply(lambda x:len(x.split()))
    data['parent_com_len']=data['parent_comment'].apply(lambda x:len(x.split()))
    data_com = tf1.transform(data['comment'])
    data_parent = tf2.transform(data['parent_comment'])
    data_author = tf3.transform(data['author'])
    data_subreddit = tf4.transform(data['subreddit'])
    X=hstack((data_com,data_parent,data_author,data_subreddit, score, ups, downs, data['parent_com_len'], data['com_len'],data['compound'], data['positive'], data['negative'], data['neutral'])).tocsr().astype('float32')

    result= xgb_model.predict(X)
    if (result==0):
        OP= 'Comment is not sarcastic'
    else:
        OP='Comment is sarcastic'
    return OP

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
