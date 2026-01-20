from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
X = X[:, 0:2]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# 切分数据：训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)

# 训练
model.fit(X_train, y_train)

# 评估
acc = model.score(X_test, y_test)
print("accuracy:", acc)
