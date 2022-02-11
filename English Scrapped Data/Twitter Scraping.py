import snscrape.modules.twitter as sntwitter
import pandas as pd


## Short Version

df_tweet_autocad = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
    'autocad lang:en near:"Toronto" within:5000km since:2020-01-01 until:2022-01-01').get_items(), 300))[['date', 'content']]



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
command_keyword = key_word+' lang:en near:"Montreal" within:1000km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
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
command_keyword = key_word+' lang:en near:"Montreal" within:1000km'+' since:'+from_date+' until:'+end_date # Define a string command for Scraper Api
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
