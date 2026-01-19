import pandas as pd

expense_df = pd.read_csv("data/expense.csv")
print(expense_df.head())
print(expense_df.shape)

category_sum = expense_df.groupby("category")["amount"].sum()
print(category_sum)

import matplotlib.pyplot as plt

category_sum.plot(kind="bar")
plt.ylabel("Total Amount")
plt.title("Total Expense by Category")
plt.show()
