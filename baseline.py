# refer to https://www.kaggle.com/pranav07/rudimentary-approach-for-ml-methods
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
pst=PorterStemmer()

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import classification_report

train_df = pd.read_csv('.data/train.tsv', sep='\t')
sub_df = pd.read_csv('./data/test.tsv', sep='\t')

# Generate unigram
list_all_words = []
for i in train_df.comment:
    words = word_tokenize(i)
    for word in words:
        pst.stem(word)
        list_all_words.append(word)

target_list = []
for i in train_df.label:
    target_list.append(i)
y = pd.Series(target_list)

count_vec = CountVectorizer(input=list_all_words, \
                            lowercase=False, \
                            min_df=2)

X_count_vec = count_vec.fit_transform(train_df.comment)
X_names = count_vec.get_feature_names()
X_count_vec = pd.DataFrame(X_count_vec.toarray(), columns=X_names)
sub_count_vec = count_vec.transform(sub_df.comment.values.astype('U'))
sub_names = count_vec.get_feature_names()
sub_count_vec = pd.DataFrame(sub_count_vec.toarray(), columns=sub_names)

print("Start training...")
# test and generate an accuracy by spliting training set
X_train_csv,X_test_csv,y_train_csv,y_test_csv=train_test_split(X_count_vec,y,test_size=0.25,random_state=42)
fit_cb=MultinomialNB()
y_mnb=fit_cb.fit(X_train_csv,y_train_csv)
y_pred_mnb=y_mnb.predict(X_test_csv)
print(metrics.accuracy_score(y_test_csv,y_pred_mnb))

print("generating submission file...")
# generate submission.csv
X_train,X_test,y_train=X_count_vec, sub_count_vec, y
fit_cb=MultinomialNB()
y_mnb=fit_cb.fit(X_train,y_train)
y_pred_mnb=y_mnb.predict(X_test)

pd.Series(y_pred_mnb).to_csv("./result/baseline_vincent.csv", header=["label"], index_label="id")



