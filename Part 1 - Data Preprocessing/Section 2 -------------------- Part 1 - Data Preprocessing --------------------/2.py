# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
df=pd.read_csv("data.csv")
x=df.iloc[:,:3].values
y=df.iloc[:,3].values
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])



from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,0] = labelencoder.fit_transform(x[:,0])



from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
traintestsplit=train_test_split()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

