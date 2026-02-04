import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("students_pass_v2.csv")

X = df[["hours_study"]].values
y = df["passed"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 只打乱训练集
rng = np.random.default_rng(42)
X_train_shuffled = rng.permutation(X_train)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train_shuffled, y_train)
acc = model.score(X_test, y_test)

print("v2 hours_study shuffled test accuracy:", acc)
