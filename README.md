# Learning ML Winter

This repository records my learning progress during the winter vacation.

## Projects

### 1. Expense Analysis
A simple data analysis project using Python and pandas.

📁 Path: `expense_analysis/`

- Read CSV data with pandas
- Group and summarize expenses by category
- Visualize results using matplotlib

## Learning Focus
- Python data analysis
- pandas
- matplotlib
- Basic data processing workflow

### 2. Iris Classification (Machine Learning)

A simple machine learning project using sklearn on the Iris dataset.

📁 Path: `iris_classification/`

- Load built-in Iris dataset from sklearn
- Split data into training and testing sets
- Train and evaluate classification models (KNN, Logistic Regression)
- Compare model performance
- Reduce feature dimensions to observe performance changes

3. Student Pass Classification (Machine Learning)

A small machine learning experiment using a real-world-style CSV dataset,
with a focus on **model evaluation and result reliability on small data**.

📁 Path: `student_pass_classification/`

**Dataset**
- Samples: 12
- Features: age, hours_study, practice_tests
- Label: passed (0 / 1)

**What I did**
- Load and preprocess CSV data with pandas and numpy
- Train a baseline model (DummyClassifier) for comparison
- Train and compare Logistic Regression and KNN
- Evaluate models using cross validation
- Test the effect of model complexity (KNN with different k values)

**Key learning points**
- Accuracy can be misleading on very small datasets
- Baseline models are essential to judge whether learning is meaningful
- Cross validation helps assess result stability
- More complex models do not always perform better on small data



