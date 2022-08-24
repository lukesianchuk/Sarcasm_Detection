import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_json("data.json", lines = True)
#print(data.loc[0:2])

# transforming columns to better variable names
data["is_sarcastic"] = data["is_sarcastic"].map({0:"Not Sarcasm",1:"Sarcasm"})

# removing unnecessary column
data = data[["headline","is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

# Using stemming to reduce input feature space
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()
x_stemmed = []
for sentence in x:
    x_stemmed.append(stemmer.stem(sentence))
x_stemmed = np.array(x_stemmed)


# Implementing CountVectorizer to transform input data into readable format
# This implements the "bag of words" format for analysis, in which 
# word order doesn't matter
cv = CountVectorizer()
X = cv.fit_transform(x_stemmed)

# Training
clf = BernoulliNB()
clf.fit(X, y)

import joblib
joblib.dump(clf, "clf.pkl")
joblib.dump(cv, "cv.pkl")

'''
inp = "mom starting to fear son's web series closest thing she will have to grandchild"
trans = cv.transform([inp]).toarray()
output = clf.predict(trans)
print(output)
output = clf.predict_proba(trans)
print(max(output[0]))
'''
