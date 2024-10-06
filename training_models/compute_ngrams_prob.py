import nltk,re,pickle,pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import Counter
import json
import os

nltk.download('punkt')

file_path = os.path.dirname(os.path.realpath(__file__))

data = pd.read_csv(file_path + '/dataset/data.csv')

unigrams = []
bigrams = []
trigrams = []

N = len(data)
for i in range(N):
    sentences = sent_tokenize(data.paragraph[i])
    for sentence in sentences:
        words = word_tokenize((re.sub(r'([.,!:|\/?@#$%^&*()_+=-])',' ',sentence)).lower())
        unigrams.extend(list(ngrams(words,1)))
        bigrams.extend(list(ngrams(words,2)))
        trigrams.extend(list(ngrams(words,3)))
        
unigram_freq = FreqDist(unigrams)
bigram_freq = FreqDist(bigrams)
trigram_freq = FreqDist(trigrams)


V = sum(unigram_freq.values())  # Total count of unigrams

bigram_sums = Counter()
for (w1, w2), freq in bigram_freq.items():
    bigram_sums[w1] += (freq + 1)

trigram_sums = Counter() 
for (w1, w2, w3), freq in trigram_freq.items():
    trigram_sums[(w1, w2)] += (freq + 1)    


#unigram probabilities
P_1 = {w: freq / V for w, freq in unigram_freq.items()}

#bigram probabilities
P_2 = {(w1, w2): (bigram_freq[(w1, w2)] + 1) / bigram_sums[w1] for (w1, w2) in bigram_freq}

#trigram probabilities
P_3 = {(w1, w2, w3): (trigram_freq[(w1, w2, w3)] + 1) / trigram_sums[(w1, w2)] for (w1, w2, w3) in trigram_freq}
       
with open('n_grams_probabilities.pkl', 'wb') as f:
    pickle.dump([P_1,P_2,P_3], f)

word_freq = dict()

for word,freq in unigram_freq.items():
    word_freq[word[0]] = freq

with open("word_freq.json", "w") as json_file:
    json.dump(word_freq, json_file)

# print(len(word_freq))



  
    