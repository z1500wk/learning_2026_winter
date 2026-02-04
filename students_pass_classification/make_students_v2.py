import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

n = 60  # 比之前多一点，但还很小

age = rng.integers(18, 25, size=n)
hours_study = rng.normal(loc=5, scale=2, size=n).clip(0)
practice_tests = rng.integers(0, 10, size=n)

# 这是“真实世界”的关键：分数 + 噪声
score = (
    0.4 * hours_study +
    0.3 * practice_tests +
    rng.normal(0, 2, size=n)
)

passed = (score > 5).astype(int)

df = pd.DataFrame({
    "age": age,
    "hours_study": hours_study,
    "practice_tests": practice_tests,
    "passed": passed
})

df.to_csv("students_pass_v2.csv", index=False)
print(df.head())
