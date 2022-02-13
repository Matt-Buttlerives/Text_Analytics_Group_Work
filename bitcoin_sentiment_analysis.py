###### Import Modules
#import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
#import itertools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
import lda
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#################################################################
###### Read Data and Preprocessing

df_surge_canada = pd.read_csv('tweets_keyword_bitcoin_surge_canada.csv')
df_surge_canada['Event'] = 'surge'
df_surge_canada['Country'] = 'Canada'


df_surge_usa = pd.read_csv('tweets_keyword_bitcoin_surge_usa.csv')
df_surge_usa['Event'] = 'surge'
df_surge_usa['Country'] = 'US'

2

df_crash_canada = pd.read_csv('tweets_keyword_bitcoin_crash_canada.csv')
df_crash_canada['Event'] = 'crash'
df_crash_canada['Country'] = 'Canada'


df_crash_usa = pd.read_csv('tweets_keyword_bitcoin_crash_usa.csv')
df_crash_usa['Event'] = 'crash'
df_crash_usa['Country'] = 'US'

df = pd.concat([df_surge_canada, df_surge_usa, df_crash_canada, df_crash_usa])
df = df[['Country','Event','Text','Datetime']]

#################################################################
###### Sentiment Analysis

SIA = SentimentIntensityAnalyzer()
sentiment_list = []
overall_sent = []

for sentence in df.Text:
    pol_sc = SIA.polarity_scores(sentence)
    sentiment_list.append(pol_sc)
    
df['Sentiment'] = sentiment_list
   
for i in sentiment_list:
    overall_sent.append(i['compound'])

df['overall_sentiment'] = overall_sent


df.to_csv("bitcoin_sentiment.csv",index=False) # Export to a csv file

###################################################################
###### Topic Modeling

df = pd.read_csv("bitcoin_sentiment.csv")

df = df.dropna()
ntopics = input('Provide the number of latent topics to be estimated: ')

word_tokenizer = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))
temp = df['Text']
def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(df['Text'])
print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
df=df.join(doc_topic)
bitcoin=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    bitcoin[topic]=df.groupby(['Country','Event'])[i].mean()

bitcoin=bitcoin.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics1=topics.transpose()
topics1.to_excel("topic_word_dist.xlsx")
bitcoin.to_excel("bitcoin_topic_dist.xlsx",index=False)

#################################################################
###### Text Classification

df = pd.read_csv("bitcoin_sentiment.csv")
df['sentiment'] = np.nan

for i in range(len(df['overall_sentiment'])):
    if df['overall_sentiment'][i] > 0:
        df['sentiment'][i] = 'Positive'
    if df['overall_sentiment'][i] < 0:
        df['sentiment'][i] = 'Negative'
    if df['overall_sentiment'][i] == 0:
        df['sentiment'][i] = 'Neutral'

tfvec = TfidfVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
tf_vectors = tfvec.fit_transform(df['Text'])

X_train, X_test, y_train, y_test = train_test_split(tf_vectors,
                                                    df['sentiment'],
                                                    test_size=0.25,
                                                    random_state=0)

y_train = np.asarray(y_train, dtype="|S6")
y_test = np.asarray(y_test, dtype="|S6")

## Trying different Naive Bayesian Classifiers

# MultinomialNB

MNB = MultinomialNB()
MNB.fit(X_train, y_train)

accuracy_MNB = accuracy_score(MNB.predict(X_test), y_test)
print('Accuracy score MNB: '+str('{:04.2f}'.format(accuracy_MNB*100))+'%')

# ComplementNB

CNB = ComplementNB()
CNB.fit(X_train, y_train)

accuracy_CNB = accuracy_score(CNB.predict(X_test), y_test)
print(str('Accuracy score CNB: '+'{:04.2f}'.format(accuracy_CNB*100))+'%')

# GaussianNB
# Note: This chunk of code takes significant amount of time to run

#GNB = GaussianNB()
#GNB.fit(X_train.todense(), y_train)

#accuracy_GNB = accuracy_score(GNB.predict(X_test), y_test)
#print(str('Accuracy score GNB: '+'{:04.2f}'.format(accuracy_GNB*100))+'%')

# BernoulliNB

BNB = BernoulliNB()
BNB.fit(X_train, y_train)

accuracy_BNB = accuracy_score(BNB.predict(X_test), y_test)
print(str('Accuracy score BNB: '+'{:04.2f}'.format(accuracy_BNB*100))+'%')

# Predicting a word with 'Positive' outcome
BNB.predict(tfvec.transform(['#bitcoin is the awesome!']))

# Predicting a word with 'Neutral' outcome
BNB.predict(tfvec.transform(['#bitcoin price is going up @ElonMusk']))

# Predicting a word 'Negative' outcome
# Note: I used a tweet example from the dataset. 
# The proportion of negative tweets are very small compared to the neutral and positive ones, 
# so the model tend to classify the words into one of those
BNB.predict(tfvec.transform(['#Bitcoin #SHIB #Ethereum IM NOT ALLOWED TO POST THIS VIDEO ANYMORE BUT I CAN QUOTE THIS TWEET LMAOOO 🤣🤣🤣YALL CANT STOP ME!!! THE ELITES WONT WIN, WE CAN WIN WE JUST NEED EVERYONE TO WAKE THE FUCK UP TO THEIR PLANS!!! BITCOIN IS THE ULTIMATE PRISON WE CANNOT ADOPT IT!!! #69 https://t.co/txzOXEWNuo']))
