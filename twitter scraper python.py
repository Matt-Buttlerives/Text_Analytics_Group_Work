#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##### INSY 669 GROUP PROJECT #####

#### Import Packages
import snscrape.modules.twitter as sntwitter
import pandas as pd


#####################################################################################################
######## BITCOIN ########
key_word = "bitcoin"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2022-01-01" # Declare a start date
end_date = '2022-01-07'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_bitcoin.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")
#####################################################################################################
##### ETHEREUM #####

key_word = "ethereum"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2022-01-01" # Declare a start date
end_date = '2022-01-07'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_ethereum.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")
#####################################################################################################
##### CRYPTO #####

key_word = "crypto"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2022-01-01" # Declare a start date
end_date = '2022-01-07'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_crypto.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")
#####################################################################################################
##### DOGECOIN #####

key_word = "doge"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2022-01-01" # Declare a start date
end_date = '2022-01-07'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_doge.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")
#####################################################################################################
##### SHIBA INU #####

key_word = "shiba"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2022-01-01" # Declare a start date
end_date = '2022-01-07'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_shiba.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")
#####################################################################################################
##### Bitcoin Surge and Crash #####

###### PRICE SURGE ######
#### Canada ####
key_word = "bitcoin"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2021-10-28" # Declare a start date
end_date = '2021-11-08'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en near:"Montreal" within:450km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_bitcoin_surge_canada.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")

#### US ####
key_word = "bitcoin"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2021-10-28" # Declare a start date
end_date = '2021-11-08'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en near:"Denver" within:1500km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_bitcoin_surge_usa.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")


############################################

##### PRICE CRASH #####
### Canada ###

key_word = "bitcoin"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2021-11-08" # Declare a start date
end_date = '2021-11-18'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en near:"Montreal" within:450km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_bitcoin_crash_canada.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")


### US ###
key_word = "bitcoin"  # Declare the key word used to search tweets
#user_name = "@mcgillu"   # Declare a user name used to search tweets
from_date = "2021-11-08" # Declare a start date
end_date = '2021-11-18'  # Declare a end date
count = 1000           # The maximum number of tweets
tweets_list_keyword = [] # A list used to store the returned results for keyword search
tweets_list_user = []    # A list used to store the retuned results for user search


#### Scraping tweets from a text search query ####
command_keyword = key_word+' lang:en near:"Denver" within:1500km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
print("Scraping data for keyword:",key_word)
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(command_keyword).get_items()):
    tweets_list_keyword.append([tweet.date,tweet.id, tweet.content, tweet.user.username, tweet.url]) # Append returned results to list
    if i>count:
        break;
# Create a dataframe from the tweets list above 
tweets_df_keyword = pd.DataFrame(tweets_list_keyword, columns=['Datetime','Tweet Id', 'Text', 'Username', 'url'])
tweets_df_keyword.to_csv("tweets_keyword_bitcoin_crash_usa.csv",index=False) # Export to a csv file
print("Scraped data have been exported to the csv file")






#################################################################
##### PRE-PROCESSING THE DATA #####

## Read Data

df = pd.read_csv('/Users/michaelchurchcarson/Desktop/MMA CLASSES/Winter 2022 P1/INSY 669 Text Analytics/Group Project/Twitter Scrapping/Scrapped Tweets/Crypto_Tweets.csv', names = ['Datetime','Tweet Id','Text', 'Username', 'url'])

############# Text Pre-processing

##### 1:Remove Punctuations
import string

# Defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# Storing the puntuation free text
df['punctuation_free']= df['Text'].apply(lambda x:remove_punctuation(x))


##### 2: Lowering Text
df['msg_lower']= df['punctuation_free'].apply(lambda x: x.lower())


##### 3: Tokenization
from nltk.tokenize import word_tokenize
df['msg_tokenized']= df['msg_lower'].apply(lambda x: word_tokenize(x))


##### 4: Removing Stop Words
import nltk

# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

# Defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

# Applying the function
df['no_stopwords']= df['msg_tokenized'].apply(lambda x:remove_stopwords(x))

##### 5: Lemmatization 

from nltk.stem import WordNetLemmatizer

# Defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


# Defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

df['msg_lemmatized']=df['no_stopwords'].apply(lambda x:lemmatizer(x))

clean_posts = df.msg_lemmatized
keywords = df.msg_lemmatized.copy()

##### 6: Frequency Distribution

from nltk.probability import FreqDist

# Importing coins.csv as coins
coins = pd.read_csv('/Users/michaelchurchcarson/Desktop/MMA CLASSES/Winter 2022 P1/INSY 669 Text Analytics/Group Project/Twitter Scrapping/Scrapped Tweets/coins.csv',names=['coin','ticker'])

# Cleaning the coins and tickers from periods, comas, and dashes
coins['coin'] = coins['coin'].str.lower()
coins['coin'] = coins['coin'].str.replace(" ","")
coins['coin'] = coins['coin'].str.replace(".","")
coins['coin'] = coins['coin'].str.replace(",","")
coins['coin'] = coins['coin'].str.replace(".","")
coins['coin'] = coins['coin'].str.replace("-","")
coins['ticker'] = coins['ticker'].str.replace(".","")
coins['ticker'] = coins['ticker'].str.replace(",","")

# Generating list of tickers
ticker_list = coins['ticker'].tolist()
ticker_list = list(dict.fromkeys(ticker_list))

# Generating list of coins
coin_list = coins['coin'].tolist()
coin_list = list(dict.fromkeys(coin_list))

#generating check list to delete all tokens not related to coins/tickers
check_list = []
check_list.extend(ticker_list)
check_list.extend(coin_list)

#empty list to contain all the words
words = []

#deleting all tokens not related to coins/tickers, finding and replacing tickers to coins
for i in range(len(clean_posts)):
    clean_posts[i] = [j for j in clean_posts[i] if j in check_list]
    for k in range(len(coins)):
        clean_posts[i] = [coins.iloc[k,0] if l == coins.iloc[k,1] else l for l in clean_posts[i]]
        clean_posts[i] = list(dict.fromkeys(clean_posts[i]))
    words.extend(clean_posts[i])

#defining an object for frequency distribution of words   

fdist = FreqDist(words)

#changing the type of df_fdist from probability.fdist to DataFrame

coin = pd.DataFrame.from_dict(fdist, orient='index')
coin.columns = ['Frequency']
coin.index.name = 'Token'
coin = coin.reset_index()

#sorting by frequency, descending

coin = coin.sort_values(by=['Frequency'], ascending=False)

#obtaining the top 5 coins
top_5 = coin.head(5)

#generating list of top 5 coins

top_5_coins = top_5['Token'].tolist()
top_5_coins = list(dict.fromkeys(top_5_coins))

#deleting tokens not in the top 5 coins
for i in range(len(clean_posts)):
    clean_posts[i] = [j for j in clean_posts[i] if j in top_5_coins]

#deleting rows with empty list

top_coins_posts = list(filter(None, clean_posts))     

import itertools

#getting all combinations of pairs
all_pairs = list(itertools.product(top_5_coins, repeat=2))

#counting the number of cooccurences of each pair
cooccurence = {pair: len([x for x in top_coins_posts if set(pair) <= set(x)]) for pair in all_pairs}

#creating an empty dataframe to contain tokens used in conjunction with each coin
keywords_count = pd.DataFrame(columns=['Token','Word_List'])
for i in range(len(top_5_coins)):
    keywords_count = keywords_count.append({'Token':top_5_coins[i],
                                            'Word_List':[top_5_coins[i],]},
                                           ignore_index=True)

#creating an empty list to contain tokens used for all cars
all_tokens = []

#assigning tokens to coins it mentioned together with and creating a list of tokens in every posts
for i in range(len(keywords)):
    for j in range(len(top_5_coins)):
        if top_5_coins[j] in keywords[i]:
            keywords_count.iloc[j,1] += list(dict.fromkeys(keywords[i]))
    all_tokens.extend(list(dict.fromkeys(keywords[i])))

#finding the frequency of each token, turning it into a dataframe and exporting it
all_tokens = [i for i in all_tokens if i not in check_list]
keyword_all = FreqDist(all_tokens)
keyword_all = keyword_all.most_common(100)
df_keyword_all = pd.DataFrame(keyword_all)
df_keyword_all.to_csv('keyword_all.csv', index=False)


#finding the frequency of each token mentioned along each coin, turning it into a dataframe and exporting it     
for i in range(len(top_5_coins)):
    keywords_count.iloc[i,1] = [k for k in keywords_count.iloc[i,1] if k not in check_list]
    globals()['keyword_'+top_5_coins[i]] = FreqDist(keywords_count.iloc[i,1])
    globals()['keyword_'+top_5_coins[i]]  = globals()['keyword_'+top_5_coins[i]].most_common(25)
    globals()['df_keyword_'+top_5_coins[i]] = pd.DataFrame(globals()['keyword_'+top_5_coins[i]])
    globals()['df_keyword_'+top_5_coins[i]].to_csv('keyword_'+top_5_coins[i]+'.csv', index=False)

#aggregating coin_association frequency

coin_association = pd.DataFrame.from_dict(cooccurence, orient='index', columns=['Joint_freq'])

#separating coin 1 and 2 of each pair
coin_association[['Coin1', 'Coin2']] = pd.DataFrame(coin_association.index.tolist(), index=coin_association.index)
coin_association.reset_index(drop=True, inplace=True)


# adding frequency of coin 1 and 2

coin_association=coin_association.merge(top_5,left_on=['Coin1'],right_on=['Token'],how='left').drop(['Token'],axis=1)

coin_association=coin_association.merge(top_5,left_on=['Coin2'],right_on=['Token'],how='left').drop(['Token'],axis=1)

#creating Lift and Dissimilarity Dataframe

coin_association['Lift']=len(df)*coin_association['Joint_freq']/(coin_association['Frequency_x']*coin_association['Frequency_y']) 
coin_association['Dissimilarity'] = 1/coin_association['Lift']
Lift_df=coin_association[['Coin1', 'Coin2', 'Lift', 'Dissimilarity']]

# reshape from long to wide in pandas python
Dissimilarity_matrix=coin_association.pivot(index='Coin1', columns='Coin2', values='Dissimilarity')

import numpy as np

#set diagonal values to zero
Dissimilarity_matrix.values[[np.arange(Dissimilarity_matrix.shape[0])]*2] = 0

from sklearn.manifold import MDS
import matplotlib.pyplot as plt

#create multidimentional scaling map
embedding = MDS(dissimilarity='precomputed', n_components=2, random_state=0)
MDS_data = embedding.fit_transform(Dissimilarity_matrix)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(122)
colors = ['r', 'g', 'b', 'c', 'm']
plt.scatter(MDS_data[:,0], MDS_data[:,1], c=colors)
plt.title('MDS Map of Cryptocurrencies')

for i, txt in enumerate(Dissimilarity_matrix.index.tolist()):
    ax.annotate(txt, (MDS_data[i,0], MDS_data[i,1]), fontsize=20)

plt.show()














































