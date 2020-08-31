import numpy as np
import pandas as pd

from dabl.models import SimpleClassifier
from dabl.explain import explain
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/processed/train_clean.csv')
target = 'Loan_Status'

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=100)

sc = SimpleClassifier()

sc.fit(X_train, y_train)

explain(sc, X_test, y_test)
