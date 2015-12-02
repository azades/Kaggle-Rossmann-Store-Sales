import pandas as pd
from sklearn import ensemble

datastr=".//data//"
df_train= pd.read_csv(datastr+'train.csv',low_memory=False)
df_test= pd.read_csv(datastr+'test.csv')
print(df_test.head())
print(df_train.head())
feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

y = df_train['Cover_Type']
test_ids = df_test['Id']

clf = ensemble.RandomForestClassifier(n_estimators = 150, n_jobs = -1)
clf_fit=clf.fit(X_train, y)
print(clf.predict(X_test).head())