import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0,'spam':1})

X_train,X_test,y_train,y_test = train_test_split(
    data['message'],data['label'],test_size=0.2)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
