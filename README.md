# Titanic Survival Prediction using Machine Learning

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. It is based on the classic Titanic dataset and demonstrates data analysis, visualization, preprocessing, and model training using Python and popular data science libraries.

## Project Overview

The goal is to build a predictive model that can determine whether a passenger survived the Titanic disaster based on various features (such as age, sex, ticket class, etc.). The project covers:
- Data loading and exploration
- Handling missing values
- Data visualization
- Feature encoding
- Model training (Logistic Regression)
- Evaluating model performance

## Dataset

The dataset used is the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) dataset, which contains information about the passengers aboard the Titanic.

**Key columns:**
- `PassengerId`
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`
- `Survived` (target)

## Features

- Data cleaning (handling missing values)
- Exploratory Data Analysis with plots (Seaborn/Matplotlib)
- Encoding categorical variables
- Logistic Regression model
- Model evaluation (accuracy score)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xainasx/Titanic---Machine-Learning-from-Disaster.git
   cd Titanic---Machine-Learning-from-Disaster
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

   Or use a Jupyter/Colab environment with these libraries.

## Usage

1. Download the Titanic dataset (`train.csv`) and place it in the root directory or update the path in the notebook.

2. Open the Jupyter notebook:

   ```bash
   jupyter notebook Titanic_Survival_Prediction_using_Machine_Learning.ipynb
   ```

   Or open it in Google Colab using [this link](https://colab.research.google.com/github/xainasx/Titanic---Machine-Learning-from-Disaster/blob/main/Titanic_Survival_Prediction_using_Machine_Learning.ipynb).

3. Run the notebook cells sequentially to explore the data, preprocess it, and train the model.

## Project Structure

```
.
├── Titanic_Survival_Prediction_using_Machine_Learning.ipynb
├── train.csv
└── README.md
```

## Results

- Visualizations show the relationship between features and survival.
- Logistic Regression model is trained and evaluated.
- Achieved accuracy depends on the preprocessing and feature engineering steps.

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## License

This project is for educational purposes.
