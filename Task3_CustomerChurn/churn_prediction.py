import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("churn.csv")

# separate features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LogisticRegression()

# train model
model.fit(X_train, y_train)

# prediction
predictions = model.predict(X_test)

# accuracy
print("Model Accuracy:", accuracy_score(y_test, predictions))
