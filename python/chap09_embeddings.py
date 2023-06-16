#!/usr/bin/env python
# coding: utf-8

# # Using Pre-trained Word Embeddings
# 
# In this notebook we will show some operations on pre-trained word embeddings to gain an intuition about them.
# 
# We will be using the pre-trained GloVe embeddings that can be found in the [official website](https://nlp.stanford.edu/projects/glove/). In particular, we will use the file `glove.6B.300d.txt` contained in this [zip file](https://nlp.stanford.edu/data/glove.6B.zip).
# 
# We will first load the GloVe embeddings using [Gensim](https://radimrehurek.com/gensim/). Specifically, we will use [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html)'s [`load_word2vec_format()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.load_word2vec_format) classmethod, which supports the original word2vec file format.
# However, there is a difference in the file formats used by GloVe and word2vec, which is a header used by word2vec to indicate the number of embeddings and dimensions stored in the file. The file that stores the GloVe embeddings doesn't have this header, so we will have to address that when loading the embeddings.
# 
# Loading the embeddings may take a little bit, so hang in there!

# In[2]:


from gensim.models import KeyedVectors

fname = "glove.6B.300d.txt"
glove = KeyedVectors.load_word2vec_format(fname, no_header=True)
glove.vectors.shape


# ## Word similarity
# 
# One attribute of word embeddings that makes them useful is the ability to compare them using cosine similarity to find how similar they are. [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html) objects provide a method called [`most_similar()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) that we can use to find the closest words to a particular word of interest. By default, [`most_similar()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) returns the 10 most similar words, but this can be changed using the `topn` parameter.
# 
# Below we test this function using a few different words.

# In[3]:


# common noun
glove.most_similar("cactus")


# In[4]:


# common noun
glove.most_similar("cake")


# In[5]:


# adjective
glove.most_similar("angry")


# In[6]:


# adverb
glove.most_similar("quickly")


# In[7]:


# preposition
glove.most_similar("between")


# In[8]:


# determiner
glove.most_similar("the")


# ## Word analogies
# 
# Another characteristic of word embeddings is their ability to solve analogy problems.
# The same [`most_similar()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) method can be used for this task, by passing two lists of words:
# a `positive` list with the words that should be added and a `negative` list with the words that should be subtracted. Using these arguments, the famous example $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$ can be executed as follows:

# In[9]:


# king - man + woman
glove.most_similar(positive=["king", "woman"], negative=["man"])


# Here are a few other interesting analogies:

# In[10]:


# car - drive + fly
glove.most_similar(positive=["car", "fly"], negative=["drive"])


# In[11]:


# berlin - germany + australia
glove.most_similar(positive=["berlin", "australia"], negative=["germany"])


# In[12]:


# england - london + baghdad
glove.most_similar(positive=["england", "baghdad"], negative=["london"])


# In[13]:


# japan - yen + peso
glove.most_similar(positive=["japan", "peso"], negative=["yen"])


# In[14]:


# best - good + tall
glove.most_similar(positive=["best", "tall"], negative=["good"])


# ## Looking under the hood
# 
# Now that we are more familiar with the [`most_similar()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) method, it is time to implement its functionality ourselves.
# But first, we need to take a look at the different parts of the [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html) object that we will need.
# Obviously, we will need the vectors themselves. They are stored in the `vectors` attribute.

# In[15]:


glove.vectors.shape


# As we can see above, `vectors` is a 2-dimensional matrix with 400,000 rows and 300 columns.
# Each row corresponds to a 300-dimensional word embedding. These embeddings are not normalized, but normalized embeddings can be obtained using the [`get_normed_vectors()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.get_normed_vectors) method.

# In[16]:


normed_vectors = glove.get_normed_vectors()
normed_vectors.shape


# Now we need to map the words in the vocabulary to rows in the `vectors` matrix, and vice versa.
# The [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html) object has the attributes `index_to_key` and `key_to_index` which are a list of words and a dictionary of words to indices, respectively.

# In[17]:


#glove.index_to_key


# In[18]:


#glove.key_to_index


# ## Word similarity from scratch
# 
# Now we have everything we need to implement a `most_similar_words()` function that takes a word, the vector matrix, the `index_to_key` list, and the `key_to_index` dictionary. This function will return the 10 most similar words to the provided word, along with their similarity scores.

# In[19]:


import numpy as np

def most_similar_words(word, vectors, index_to_key, key_to_index, topn=10):
    # retrieve word_id corresponding to given word
    word_id = key_to_index[word]
    # retrieve embedding for given word
    emb = vectors[word_id]
    # calculate similarities to all words in out vocabulary
    similarities = vectors @ emb
    # get word_ids in ascending order with respect to similarity score
    ids_ascending = similarities.argsort()
    # reverse word_ids
    ids_descending = ids_ascending[::-1]
    # get boolean array with element corresponding to word_id set to false
    mask = ids_descending != word_id
    # obtain new array of indices that doesn't contain word_id
    # (otherwise the most similar word to the argument would be the argument itself)
    ids_descending = ids_descending[mask]
    # get topn word_ids
    top_ids = ids_descending[:topn]
    # retrieve topn words with their corresponding similarity score
    top_words = [(index_to_key[i], similarities[i]) for i in top_ids]
    # return results
    return top_words


# Now let's try the same example that we used above: the most similar words to "cactus".

# In[20]:


vectors = glove.get_normed_vectors()
index_to_key = glove.index_to_key
key_to_index = glove.key_to_index
most_similar_words("cactus", vectors, index_to_key, key_to_index)


# ## Analogies from scratch
# 
# The `most_similar_words()` function behaves as expected. Now let's implement a function to perform the analogy task. We will give it the very creative name `analogy`. This function will get two lists of words (one for positive words and one for negative words), just like the [`most_similar()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar) method we discussed above.

# In[21]:


from numpy.linalg import norm

def analogy(positive, negative, vectors, index_to_key, key_to_index, topn=10):
    # find ids for positive and negative words
    pos_ids = [key_to_index[w] for w in positive]
    neg_ids = [key_to_index[w] for w in negative]
    given_word_ids = pos_ids + neg_ids
    # get embeddings for positive and negative words
    pos_emb = vectors[pos_ids].sum(axis=0)
    neg_emb = vectors[neg_ids].sum(axis=0)
    # get embedding for analogy
    emb = pos_emb - neg_emb
    # normalize embedding
    emb = emb / norm(emb)
    # calculate similarities to all words in out vocabulary
    similarities = vectors @ emb
    # get word_ids in ascending order with respect to similarity score
    ids_ascending = similarities.argsort()
    # reverse word_ids
    ids_descending = ids_ascending[::-1]
    # get boolean array with element corresponding to any of given_word_ids set to false
    given_words_mask = np.isin(ids_descending, given_word_ids, invert=True)
    # obtain new array of indices that doesn't contain any of the given_word_ids
    ids_descending = ids_descending[given_words_mask]
    # get topn word_ids
    top_ids = ids_descending[:topn]
    # retrieve topn words with their corresponding similarity score
    top_words = [(index_to_key[i], similarities[i]) for i in top_ids]
    # return results
    return top_words


# Let's try this function with the $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$ example we discussed above.

# In[22]:


positive = ["king", "woman"]
negative = ["man"]
vectors = glove.get_normed_vectors()
index_to_key = glove.index_to_key
key_to_index = glove.key_to_index
analogy(positive, negative, vectors, index_to_key, key_to_index)


# In[ ]:




