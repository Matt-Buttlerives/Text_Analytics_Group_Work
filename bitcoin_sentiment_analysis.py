###### Import Modules
#import snscrape.modules.twitter as sntwitter
import pandas as pd
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

#################################################################
###### Read Data and Preprocessing

df_surge_canada = pd.read_csv('tweets_keyword_bitcoin_surge_canada.csv')
df_surge_canada['Event'] = 'surge'
df_surge_canada['Country'] = 'Canada'


df_surge_usa = pd.read_csv('tweets_keyword_bitcoin_surge_usa.csv')
df_surge_usa['Event'] = 'surge'
df_surge_usa['Country'] = 'US'



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

