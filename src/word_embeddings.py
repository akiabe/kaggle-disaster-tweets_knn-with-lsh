import re
import string

import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def process_tweet(tweet):
    """
    :param tweets: a strings containing a tweet
    :return freqs: a list of word containing the processed tweet
    """
    # remove url
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)

    # remove retweet text "RT"
    tweet = re.sub(r'^RT+', '', tweet)

    # remove hashtags
    tweet = re.sub(r'#', '', tweet)

    # instantiate tokenizer class
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )

    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet)

    # import the english stop words list from NLTK
    stopwords_english = stopwords.words('english')

    # instantiate stemming class
    stemmer = PorterStemmer()

    # create empty list to store the clean tweets
    tweets_clean = []

    # remove stop words and punctuations
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def get_document_embedding(tweet, en_embeddings):
    """
    :param tweet: a string
    :param en_embeddings: a dictionary of word embeddings
    :return doc_embedding: sum of all word embeddings in the tweet
    """
    doc_embedding = np.zeros(300)

    # process the document into a list of words
    processed_doc = process_tweet(tweet)

    # loop through the process document
    for word in processed_doc:
        # add the word embedding
        doc_embedding += en_embeddings.get(word, 0)

    return doc_embedding

def get_document_vecs(all_docs, en_embeddings):
    """
    :param all_docs: list of strings, all tweets in dataset
    :param en_embeddings: dictionary with words as the keys and their embeddings as the values
    :return document_vec_matrix: matrix of tweet embeddings
    :return ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values
    """
    # the dictionary's key is an idx that identified a specific tweet
    # the value is the document embedding for that document
    ind2Doc_dict = {}

    # store document vector
    document_vec_l = []

    for i, doc in enumerate(all_docs):
        # get the document embedding of the tweet
        doc_embedding = get_document_embedding(doc, en_embeddings)

        # save the document embedding into the ind2Tweet dictionary at idx i
        ind2Doc_dict[i] = doc_embedding

        # append the document embedding to the list of document vectors
        document_vec_l.append(doc_embedding)

    # convert the list of document vectors into a 2D array
    document_vec_matrix = np.vstack(document_vec_l)

    return  document_vec_matrix, ind2Doc_dict




















