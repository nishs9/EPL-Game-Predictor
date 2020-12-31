import os.path
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load in training data
currDirectory = os.path.abspath(os.path.dirname(__file__))
raw_data = pd.read_csv(os.path.join(currDirectory,"../data/E0-2018-2019.csv"))
raw_data2 = pd.read_csv(os.path.join(currDirectory,"../data/E0-2019-2020.csv"))
raw_data = pd.concat([raw_data,raw_data2],axis=0)

#Cleaning up and transforming the dataframe for use in the DT model
results = raw_data[["FTR"]]
matches = raw_data[["HomeTeam","AwayTeam","B365H","B365D","B365A","BWH","BWD","BWA"]]
home = pd.get_dummies(matches["HomeTeam"],prefix='HomeTeam')
away = pd.get_dummies(matches["AwayTeam"],prefix='AwayTeam')
B365H = matches["B365H"]
B365D = matches["B365D"]
B365A = matches["B365A"]
BWH = matches["BWH"]
BWD = matches["BWD"]
BWA = matches["BWA"]
matches = pd.concat([home,away,B365H,B365D,B365A,BWH,BWD,BWA],axis=1)
#Adding in the teams that were promoted for the 2020 season
matches["HomeTeam_Leeds"] = 0
matches["HomeTeam_West Brom"] = 0
matches["AwayTeam_Leeds"] = 0
matches["AwayTeam_West Brom"] = 0

#Distribution of labels
vizDF = raw_data[["FTR","HomeTeam","AwayTeam","B365H","B365D","B365A","BWH","BWD","BWA"]]
xAxis = vizDF["B365A"].tolist()
yAxis = vizDF["FTR"].tolist()
plt.scatter(xAxis,yAxis)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(matches, results, test_size=0.2, shuffle=True)
clf = DecisionTreeClassifier(max_depth=3,criterion='entropy',splitter="random",min_samples_split=7,min_samples_leaf=3,class_weight={"H":0.44,"A":0.33,"D":0.22})
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
#print("\nMinSampleSplit: " + str(i) + "\nAccuracy:",acc)
print(acc)