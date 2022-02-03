###########################################################################################
###### Text Extraction

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup # documentation available at : www.crummy.com/software/BeautifulSoup/bs4/doc/
import requests # sends http requests and access the page : docs.python-requests.org/en/latest/
import csv # creates the output csv file
#import unicodedata # works with string encoding of the data

entries = []
entry = []
urlnumber = 290 # Give the page number to start with


while urlnumber < 396: # Give the page number to end with

    url = 'https://forums.edmunds.com/discussion/7526/general/x/midsize-sedans-2-0/p%d' % (urlnumber,) # Give the url of the forum, excluding the page number in the hyperlink

    try:
        r = requests.get(url, timeout = 10) # Sending a request to access the page
    except Exception as e:
        print("Error message:",e)
        break;

    data = r.text
    
    soup = BeautifulSoup(data, 'lxml') # Getting the page source into the soup
    
    for div in soup.find_all('div'):
        entry = []
        if(div.get('class') != None and div.get('class')[0] == 'Comment'): # A single post is referred to as a comment. Each comment is a block denoted in a div tag which has a class called comment.
            ps = div.find_all('p') # gets all the tags called p to a variable ps
            aas = div.find_all('a') # gets all the tags called a to a variable aas
            spans = div.find_all('span')
            times = div.find_all('time') # used to extract the time tag which gives the iDate of the post

            concat_str = ''
            for str in aas[1].contents: # prints the contents that is between the tag start and end
                if str != "<br>" or str != "<br/>": # breaks in post which we need to work around
                    concat_str = (concat_str + ' '+ str).encode("utf-8").strip() # the format extracted is a unicode - we need a uniform structure to work with the strings
            entry.append(concat_str)

            concat_str = ''
            for str in times[0].contents:
                if str != "<br>" or str != "<br/>":
                    concat_str = (concat_str + ' '+ str).encode('iso-8859-1').strip()
            entry.append(concat_str)

            for div in div.find_all('div'):
                if (div.get('class') != None and div.get('class')[0] == 'Message'): # extracting the div tag with the class attribute as message
                    blockquotes = []
                    x = div.get_text()
                    for bl in div.find_all('blockquote'):
                        blockquotes.append(bl.get_text()) # block quote is used to get the quote made by a person. get_text helps to eliminate the hyperlinks and pulls out only the data.
                        bl.decompose()
                    # Encoding the text to ascii code by replacing the non-ascii characters
                    ascii_encoding = div.get_text().replace("\n"," ").replace("<br/>","").encode('ascii','replace')
                    # Convert the ASCII encoding to Latin1 encoding
                    latin1_encoding = ascii_encoding.decode('ascii').encode('iso-8859-1')
                    # Append the encoding bytes to output list
                    entry.append(latin1_encoding)

                    for bl in blockquotes:
                        ascii_encoding = bl.replace("\n"," ").replace("<br/>","").encode('ascii','replace')
                        latin1_encoding = ascii_encoding.decode('ascii').encode('iso-8859-1')
                        entry.append(latin1_encoding)

            entries.append(entry)
            
    urlnumber += 1

# Convert a list of byte to list a of string     
stringlist=[[x.decode('iso-8859-1') for x in entry] for entry in entries]
# Save the list to a csv file
with open('edmunds_extraction.csv', 'w') as output:
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerows(stringlist)

print ("Wrote to edmunds_extraction.csv")

###########################################################################################

######## Pre-processing

## Read Data

import pandas as pd
df = pd.read_csv('edmunds_extraction.csv', names = ['user_id','date','text'])

############# Text Pre-processing

##### 1:Remove Punctuations
import string

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#storing the puntuation free text
df['punctuation_free']= df['text'].apply(lambda x:remove_punctuation(x))


##### 2: Lowering Text
df['msg_lower']= df['punctuation_free'].apply(lambda x: x.lower())


##### 3: Tokenization
from nltk.tokenize import word_tokenize
df['msg_tokenized']= df['msg_lower'].apply(lambda x: word_tokenize(x))


##### 4: Removing Stop Words
import nltk

#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
df['no_stopwords']= df['msg_tokenized'].apply(lambda x:remove_stopwords(x))


##### 5: Lemmatization 

from nltk.stem import WordNetLemmatizer

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

df['msg_lemmatized']=df['no_stopwords'].apply(lambda x:lemmatizer(x))

clean_posts = df.msg_lemmatized
keywords = df.msg_lemmatized.copy()

##### 6: Frequency Distribution

from nltk.probability import FreqDist

#importing models.csv as models
models = pd.read_csv(r'models.csv',names=['brand','model'])

#cleaning the brands and models from periods, comas, and dashes
models['model'] = models['model'].str.lower()
models['model'] = models['model'].str.replace(" ","")
models['model'] = models['model'].str.replace(".","")
models['model'] = models['model'].str.replace(",","")
models['model'] = models['model'].str.replace(".","")
models['model'] = models['model'].str.replace("-","")
models['brand'] = models['brand'].str.replace(".","")
models['brand'] = models['brand'].str.replace(",","")

#adding brands-models that are not included in models
models.iloc[-1] = ['hyundai','difhyundairent']
models.iloc[-1] = ['hyundai','sx']
models.iloc[-1] = ['bmw','i4']
models.iloc[-1] = ['buick','verano']
models.iloc[-1] = ['nissan','malibu']
models.iloc[-1] = ['nissan','malibus']
models.iloc[-1] = ['cadenza','kia']
models.iloc[-1] = ['hyundai','sonataoptima']
models.iloc[-1] = ['infiniti','infinity']
models.iloc[-1] = ['sonota','hyundai']
models.iloc[-1] = ['kia','k5']
models.iloc[-1] = ['toyota','rav4']
models.iloc[-1] = ['hyundai','hyudai']
models.iloc[-1] = ['bmw','i5']
models.iloc[-1] = ['chevrolet','z24']

#remove unnecessary rows from models
models = models[models['brand'] != 'hyndai kia']
models = models[models['brand'] != 'car']
models = models[models['brand'] != 'sedan']
models = models[models['brand'] != 'seat']
models = models[models['brand'] != 'problem']

#generating list of brands
brand_list = models['brand'].tolist()
brand_list = list(dict.fromkeys(brand_list))

#generating list of models
model_list = models['model'].tolist()
model_list = list(dict.fromkeys(model_list))

#generating check list to delete all tokens not related to brands / models
check_list = []
check_list.extend(brand_list)
check_list.extend(model_list)

#empty list to contain all the words
words = []

#deleting all tokens not related to brands/models, finding and replacing models to brands
for i in range(len(clean_posts)):
    clean_posts[i] = [j for j in clean_posts[i] if j in check_list]
    for k in range(len(models)):
        clean_posts[i] = [models.iloc[k,0] if l == models.iloc[k,1] else l for l in clean_posts[i]]
        clean_posts[i] = list(dict.fromkeys(clean_posts[i]))
    words.extend(clean_posts[i])
 
#defining an object for frequency distribution of words   
fdist = FreqDist(words)

#changing the type of df_fdist from probability.fdist to DataFrame
brand = pd.DataFrame.from_dict(fdist, orient='index')
brand.columns = ['Frequency']
brand.index.name = 'Token'
brand = brand.reset_index()

#sorting by frequency, descending
brand = brand.sort_values(by=['Frequency'], ascending=False)

#obtaining the top 10 brands
top_10 = brand.head(10)

#generating list of top 10 brands
top_10_brands = top_10['Token'].tolist()
top_10_brands = list(dict.fromkeys(top_10_brands))

#deleting tokens not in the top 10 brands
for i in range(len(clean_posts)):
    clean_posts[i] = [j for j in clean_posts[i] if j in top_10_brands]

#deleting rows with empty list
top_brand_posts = list(filter(None, clean_posts))     

import itertools

#getting all combinations of pairs
all_pairs = list(itertools.combinations(top_10_brands, 2))

#counting the number of cooccurences of each pair
cooccurence = {pair: len([x for x in top_brand_posts if set(pair) <= set(x)]) for pair in all_pairs}

#creating an empty dataframe to contain tokens used in conjunction with each brand
keywords_count = pd.DataFrame(columns=['Token','Word_List'])
for i in range(len(top_10_brands)):
    keywords_count = keywords_count.append({'Token':top_10_brands[i],
                                            'Word_List':[top_10_brands[i],]},
                                           ignore_index=True)

#creating an empty list to contain tokens used for all cars
all_tokens = []

#assigning tokens to brands it mentioned together with and creating a list of tokens in every posts
for i in range(len(keywords)):
    for j in range(len(top_10_brands)):
        if top_10_brands[j] in keywords[i]:
            keywords_count.iloc[j,1] += list(dict.fromkeys(keywords[i]))
    all_tokens.extend(list(dict.fromkeys(keywords[i])))

#finding the frequency of each token, turning it into a dataframe and exporting it
all_tokens = [i for i in all_tokens if i not in check_list]
keyword_all = FreqDist(all_tokens)
keyword_all = keyword_all.most_common(100)
df_keyword_all = pd.DataFrame(keyword_all)
df_keyword_all.to_csv('keyword_all.csv', index=False)
       
#finding the frequency of each token mentioned along each brand, turning it into a dataframe and exporting it     
for i in range(len(top_10_brands)):
    keywords_count.iloc[i,1] = [k for k in keywords_count.iloc[i,1] if k not in check_list]
    globals()['keyword_'+top_10_brands[i]] = FreqDist(keywords_count.iloc[i,1])
    globals()['keyword_'+top_10_brands[i]]  = globals()['keyword_'+top_10_brands[i]].most_common(25)
    globals()['df_keyword_'+top_10_brands[i]] = pd.DataFrame(globals()['keyword_'+top_10_brands[i]])
    globals()['df_keyword_'+top_10_brands[i]].to_csv('keyword_'+top_10_brands[i]+'.csv', index=False)

#aggregating brand_association frequency
brand_association = pd.DataFrame.from_dict(cooccurence, orient='index', columns=['Joint_freq'])

#separating brand 1 and 2 of each pair
brand_association[['Brand1', 'Brand2']] = pd.DataFrame(brand_association.index.tolist(), index=brand_association.index)
brand_association.reset_index(drop=True, inplace=True)

#adding frequency of brand 1 and 2
brand_association=brand_association.merge(top_10_brands,left_on=['Brand1'],right_on=['Token'],how='left').drop(['Token'],axis=1)
brand_association=brand_association.merge(top_10_brands,left_on=['Brand2'],right_on=['Token'],how='left').drop(['Token'],axis=1)

#creating Lift and Dissimilarity Dataframe
brand_association['Lift']=len(df)*brand_association['Joint_freq']/(brand_association['Frequency_x']*brand_association['Frequency_y']) 
brand_association['Dissimilarity'] = 1/brand_association['Lift']
Lift_df=brand_association[['Brand1', 'Brand2', 'Lift', 'Dissimilarity']]

# reshape from long to wide in pandas python
Dissimilarity_matrix=brand_association.pivot(index='Brand1', columns='Brand2', values='Dissimilarity')

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
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'b', 'g', 'b']
plt.scatter(MDS_data[:,0], MDS_data[:,1], c=colors)
plt.title('MDS Map of Car Brands')

for i, txt in enumerate(Dissimilarity_matrix.index.tolist()):
    ax.annotate(txt, (MDS_data[i,0], MDS_data[i,1]), fontsize=20)

plt.show()