import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 读数据
df = pd.read_csv("students_pass_classification/students_pass.csv")

X = df[["practice_tests"]].values
y = df["passed"].values

# 固定一次切分（为了好理解）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 👇 关键操作：只在训练集里打乱
import numpy as np
rng = np.random.default_rng(42)
X_train_shuffled = rng.permutation(X_train)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train_shuffled, y_train)

acc = model.score(X_test, y_test)
print("test accuracy after shuffle:", acc)
