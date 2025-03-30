import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
submission_df = pd.DataFrame()

#this prints the datatypes of the fields if imported properly
print(train_df.head())

id = "seatid" #set to the column representing id
target = "sold_first_hour" #set to the target column
categorical = ["seat_type", "speed_category"]  #list of non-numerical columns

#list of all columns that arent categorical, or the id/target
numerical = [c for c in train_df.columns if c not in categorical and c != id and c != target]

#use label encoder on each categorical column
for c in categorical:
    le = LabelEncoder()
    combined = pd.concat([train_df[c], test_df[c]])
    le.fit(combined)
    train_df[c] = le.transform(train_df[c])
    test_df[c] = le.transform(test_df[c])

print(train_df.head())

#use standard scaling on numerical values here
scaler = StandardScaler()
train_df[numerical] = scaler.fit_transform(train_df[numerical])
test_df[numerical] = scaler.transform(test_df[numerical])

print(train_df.head())

#define dataframes for model training here (X_train, y_train, X_test)
X_train = train_df.drop([id, target], axis=1)
y_train = train_df[target]
X_test = test_df.drop([id], axis=1)


#fit and predict knn here, creating a target column in test_df
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
test_df[target] = knn.predict(X_test)

submission = test_df[["seatid", "sold_first_hour"]]
submission.to_csv("submission.csv", index=False)

