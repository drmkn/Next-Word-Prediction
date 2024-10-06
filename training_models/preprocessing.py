import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
import os

file_path = os.path.dirname(os.path.realpath(__file__))

essay_data = pd.read_csv(file_path + '/dataset/ielts_writing_dataset.csv')
essays = essay_data.Essay[:600]
nips_papers = pd.read_csv(file_path + '/dataset/Papers.csv')
nips_paragraphs = nips_papers.Title + ". " + nips_papers.Abstract

paragraphs = pd.concat([essays, nips_paragraphs],ignore_index=True)
paragraphs.reset_index()
paragraphs = pd.DataFrame(paragraphs)
paragraphs.columns = ['paragraph']

paragraphs.to_csv('./dataset/data.csv', header=True, index = False)
# print(paragraphs)
# paragraphs = paragraphs.apply(lambda para : sent_tokenize(para))
# train_data, test_data = train_test_split(paragraphs,test_size = 0.1)


# train_data.to_csv('train_data.csv', header=True, index=False)
# test_data.to_csv('test_data.csv', header=True, index=False)






