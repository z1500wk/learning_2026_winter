import pandas as pd

df = pd.read_csv("students_pass_classification/students_pass.csv")
print(df)

X = df[["age", "hours_study", "practice_tests"]].values
y = df["passed"].values

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("accuracy:", acc)

from sklearn.model_selection import cross_val_score

model = LogisticRegression(max_iter=200)

scores = cross_val_score(model, X, y, cv=4)

print("cross val scores:", scores)
print("mean accuracy:", scores.mean())

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(dummy, X, y, cv=4)

print("dummy scores:", dummy_scores)
print("dummy mean:", dummy_scores.mean())

print(df["passed"].value_counts())
print(df["passed"].value_counts(normalize=True))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn_scores = cross_val_score(knn, X, y, cv=4)

print("knn scores:", knn_scores)
print("knn mean:", knn_scores.mean())
