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

### 3. Student Pass Classification (Machine Learning)

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


**What I did (v1)**
• Load and preprocess CSV data with pandas and numpy
• Train a baseline model (DummyClassifier) for comparison
• Train and compare Logistic Regression and KNN
• Test single-feature models instead of only multi-feature models
• Observe that very high accuracy (e.g. 1.0) can be misleading on small datasets

**Key observations (v1)**
• Some single features achieved perfect accuracy (1.0) on small data
• After shuffling features only in the training set, model performance collapsed
• This indicates that certain features were effectively acting as label proxies
• Learned to distinguish between:
• genuinely useful features
• and features that leak answer information
￼
**3.1 Feature Engineering & Data Leakage Check**
An extension of the student pass experiment, focusing on how to judge whether a feature is truly useful.

📁 Path: students_pass_classification/

**Core idea**
Instead of asking “Which model gets the highest score?”,
I focused on “Which features provide stable, non-cheating signal?”.
*Experiments*
• Single-feature classification (hours_study, practice_tests)
• Baseline comparison with DummyClassifier
• Train/test split for intuitive understanding
• Shuffle feature values only inside the training set as a sanity check

*Findings*
• Features that directly encode or strongly bind to the label:
• achieve unrealistically high accuracy
• collapse immediately after training-set shuffling
• More realistic features:
• improve accuracy probabilistically
• do not break instantly when disturbed

**3.2 Revised Dataset (v2)**
To better simulate a real-world scenario, I generated a second version of the dataset.

📁 Path: students_pass_classification/students_pass_v2.csv

*Dataset (v2)*
• Generated using multiple features + random noise
• Label is not determined by any single feature
• Better reflects uncertainty and variability in real data

*Observations (v2)*
• Single features such as hours_study still improve performance
• Accuracy is high but not perfect
• Shuffling the training set does not instantly destroy performance
• Demonstrates the difference between:
• data leakage
• and probabilistic but valid feature signals

**What I learned**
• High accuracy on small data can be a red flag
• Baseline models are essential for interpretation
• Feature usefulness should be tested, not assumed
• Data leakage can be detected with simple, controlled experiments
• Stable average behavior matters more than a single lucky split

