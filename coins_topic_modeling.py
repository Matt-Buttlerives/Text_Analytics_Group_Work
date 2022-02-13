###### Import Modules
import lda
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
import nltk
nltk.download('vader_lexicon')

#################################################################
###### Read Data and Preprocessing

df_bitcoin = pd.read_csv(r'tweets_keyword_bitcoin.csv')
df_bitcoin['Coin'] = 'Bitcoin'


df_crypto = pd.read_csv(r'tweets_keyword_CRO.csv')
df_crypto['Coin'] = 'CRO'



df_doge = pd.read_csv(r'tweets_keyword_doge.csv')
df_doge['Coin'] = 'Doge'


df_ethereum = pd.read_csv(r'tweets_keyword_ethereum.csv')
df_ethereum['Coin'] = 'Ethereum'

df_shiba = pd.read_csv(r'tweets_keyword_shiba.csv')
df_shiba['Coin'] = 'Shiba'


df = pd.concat([df_bitcoin, df_crypto, df_doge, df_ethereum, df_shiba])
df = df[['Coin','Text','Datetime']]
df.reset_index(inplace=True)
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


df.to_csv("sentiment_5_coins.csv",index=False) # Export to a csv file


###################################################################


###################################################################
###### Topic Modeling

# df = pd.read_csv("sentiment_5_coins.csv")

df = df.dropna()
ntopics = input('Provide the number of latent topics to be estimated: ')

############# Text Pre-processing

##### 1:Remove Punctuations
import string

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
df['punctuation_free']= df['Text'].apply(lambda x:remove_punctuation(x))


##### 2: Lowering Text
df['msg_lower']= df['punctuation_free'].apply(lambda x: x.lower())


##### 3: Tokenization
from nltk.tokenize import word_tokenize
df['msg_tokenized']= df['msg_lower'].apply(lambda x: word_tokenize(x))


##### 4: Removing Stop Words
import nltk

#Stop words present in the library
stop_words = nltk.corpus.stopwords.words('english')
#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stop_words]
    return output

#applying the function
df['no_stopwords']= df['msg_tokenized'].apply(lambda x:remove_stopwords(x))


##### 5: Lemmatization 

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

df['msg_lemmatized']=df['no_stopwords'].apply(lambda x:lemmatizer(x))

clean_posts = df.msg_lemmatized

#empty list to contain all the words
words = []

#adding words to list
for i in range(len(clean_posts)):
    words.extend([j for j in clean_posts[i]])

#defining an object for frequency distribution of words   
fdist = FreqDist(words)

#changing the type of df_fdist from probability.fdist to DataFrame
words_df = pd.DataFrame.from_dict(fdist, orient='index')
words_df.columns = ['Frequency']
words_df.index.name = 'Token'
words_df = words_df.reset_index()

#sorting by frequency, descending
words_df = words_df.sort_values(by=['Frequency'], ascending=False)

#obtaining the high frequent words
top_words = words_df.head(200)

#generating list of top words
top_words_list = top_words['Token'].tolist()
top_words_list = list(dict.fromkeys(top_words_list))

coin_names = ['bitcoin', 'btc', 'cro', 'dogecoin', 'doge', 'ethereum', 'eth', 'shiba', 'inu', 'shib']

#Removing name of coins from high frequent words
for item in coin_names:
    if item in top_words_list: top_words_list.remove(item)

#deleting tokens not in the top words and merging the words
for i in range(len(clean_posts)):
    clean_posts[i] = [j for j in clean_posts[i] if j in top_words_list]
    clean_posts[i] = " ".join(clean_posts[i])

#adding clean_posts to df
df = pd.concat([df, clean_posts.to_frame(name="top_words")], axis=1)

# topic modeling
vec_words = CountVectorizer(decode_error='ignore')
total_features_words = vec_words.fit_transform(df['top_words'])
print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
df=df.join(doc_topic)
coin=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    coin[topic]=df.groupby(['Coin'])[i].mean()

coin=coin.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics1=topics.transpose()
topics1.to_excel("topic_word_dist.xlsx")
coin.to_excel("coin_topic_dist.xlsx",index=False)

# =============================================================================
# 
# #################################################################
# ###### Text Classification
# 
# df = pd.read_csv("sentiment_5_coins.csv")
# 
# tfvec = TfidfVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
# tf_vectors = tfvec.fit_transform(df['Text'])
# 
# X_train, X_test, y_train, y_test = train_test_split(tf_vectors,
#                                                     df['overall_sentiment'],
#                                                     test_size=0.25,
#                                                     random_state=0)
# 
# y_train = np.asarray(y_train, dtype="|S6")
# y_test = np.asarray(y_test, dtype="|S6")
# 
# ## Trying different Naive Bayesian Classifiers
# 
# # MultinomialNB
# 
# MNB = MultinomialNB()
# MNB.fit(X_train, y_train)
# 
# accuracy_MNB = accuracy_score(MNB.predict(X_test), y_test)
# print('Accuracy score MNB: '+str('{:04.2f}'.format(accuracy_MNB*100))+'%')
# 
# # ComplementNB
# 
# CNB = ComplementNB()
# CNB.fit(X_train, y_train)
# 
# accuracy_CNB = accuracy_score(CNB.predict(X_test), y_test)
# print(str('Accuracy score CNB: '+'{:04.2f}'.format(accuracy_CNB*100))+'%')
# 
# # GaussianNB
# # Note: This chunk of code takes significant amount of time to run
# 
# #GNB = GaussianNB()
# #GNB.fit(X_train.todense(), y_train)
# 
# #accuracy_GNB = accuracy_score(GNB.predict(X_test), y_test)
# #print(str('Accuracy score GNB: '+'{:04.2f}'.format(accuracy_GNB*100))+'%')
# 
# # BernoulliNB
# 
# BNB = BernoulliNB()
# BNB.fit(X_train, y_train)
# 
# accuracy_BNB = accuracy_score(BNB.predict(X_test), y_test)
# print(str('Accuracy score BNB: '+'{:04.2f}'.format(accuracy_BNB*100))+'%')
# 
# =============================================================================
