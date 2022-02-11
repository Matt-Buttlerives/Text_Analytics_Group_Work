###### Import Modules
import snscrape.modules.twitter as sntwitter
import pandas as pd
import itertools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

#################################################################
###### Read Data and Preprocessing

df_surge_canada = pd.read_csv(r'C:\Users\Asus\Desktop\Text Analytics\Group Project\Scrapped Data\tweets_keyword_bitcoin_surge_canada.csv')
df_surge_canada['Event'] = 'surge'
df_surge_canada['Country'] = 'Canada'


df_surge_usa = pd.read_csv(r'C:\Users\Asus\Desktop\Text Analytics\Group Project\Scrapped Data\tweets_keyword_bitcoin_surge_usa.csv')
df_surge_usa['Event'] = 'surge'
df_surge_usa['Country'] = 'US'



df_crash_canada = pd.read_csv(r'C:\Users\Asus\Desktop\Text Analytics\Group Project\Scrapped Data\tweets_keyword_bitcoin_crash_canada.csv')
df_crash_canada['Event'] = 'crash'
df_crash_canada['Country'] = 'Canada'


df_crash_usa = pd.read_csv(r'C:\Users\Asus\Desktop\Text Analytics\Group Project\Scrapped Data\tweets_keyword_bitcoin_crash_usa.csv')
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

