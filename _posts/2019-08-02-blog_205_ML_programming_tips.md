---
layout: post
title:  "Machine Learning Programming Tips, Best Practices"
date:   2019-08-02 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

# How to apply same data preprocessing steps to train and test data while working with scikit-learn?

The general idea is save the preprocessing steps in `.pkl` file using `joblib` and reuse them during prediction. This will ensure consistency. If you are using `scikit learn`, then there is an easy way to club preprocessing and modelling in same `object`.

Use `Pipeline`.

Example:

Say in your data you have both numerical and categorical columns. And you need to apply some processing on that and you also want to make sure to apply them during the prediction phase. Also both training and prediction phases are two different pipeline. In such situation you can apply something like this:

```
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = data[feat_cols]
y = data["OUTCOME"]

numeric_features = feat_cols
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=42, stratify=y)

clf.fit(X_train, y_train)

clf.predict_proba(X_test)

model_file = "../model/model_randomforest.pkl"
joblib.dump(clf, model_file)

```

You can also add preprocessing steps for categorical column as well.

**Reference:**

- [Datascience stackexchange](https://datascience.stackexchange.com/questions/48026/how-to-correctly-apply-the-same-data-transformation-used-on-the-training-datas)


----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>