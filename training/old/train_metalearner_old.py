from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load the iris dataset and split it into train and test sets
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# define the base models
level0 = list()
level0.append(('svm', SVC()))
level0.append(('cart', DecisionTreeClassifier()))

# define meta learner model
level1 = LogisticRegression()

# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# fit the model on all available data
model.fit(X_train, y_train)

# make a prediction for one example
yhat = model.predict(X_test)

print("Test Accuracy: ", accuracy_score(y_test, yhat))