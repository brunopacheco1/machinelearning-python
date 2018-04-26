import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Import the dataset
dataSet = pd.read_csv("../data/restaurant_reviews.tsv", delimiter = "\t", quoting = 3)

# Cleaning the dataset
ps = PorterStemmer()
corpus = []

for i in range(0, 1000):
    review = dataSet["Review"][i]
    review = re.sub("[^a-zA-Z]", " ", dataSet["Review"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
# Creating the bag of words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataSet.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting
y_predicted = classifier.predict(X_test)

# Metrics
confusion_matrix = confusion_matrix(y_test, y_predicted)

accuracy = accuracy_score(y_test, y_predicted)

precision = precision_score(y_test, y_predicted)

recall = recall_score(y_test, y_predicted)

f1Score = f1_score(y_test, y_predicted)

print("Accuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}".format(accuracy, precision, recall, f1Score))