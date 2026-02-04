import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 读数据
df = pd.read_csv("students_pass_classification/students_pass.csv")
y = df["passed"].values

# baseline：什么都不看，永远猜“最多的那一类”
dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(
    dummy,
    df[["age"]].values,  # 随便给个 X，占位
    y,
    cv=4
)

print("baseline mean:", dummy_scores.mean())

# 把 practice_tests 随机打乱
shuffled = df["practice_tests"].sample(frac=1, random_state=42).values
X = shuffled.reshape(-1, 1)


# 测试一个特征：practice_tests
X = df[["practice_tests"]].values

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

scores = cross_val_score(model, X, y, cv=4)

print("practice_tests mean:", scores.mean())
