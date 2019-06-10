import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.datasets import make_classification
from sklearn import preprocessing as prep

testx, testy=make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

i_Features=np.load("600839_i.npy")
o_UpDown=np.load("600839_o.npy")

i_Features_norm=prep.scale(i_Features)
i_Train = i_Features_norm[0:int(len(i_Features)*1), :]
i_Test = i_Features_norm[int(len(i_Features)*1-200):len(i_Features), :]
o_Train = o_UpDown[0:len(i_Train)-1]
o_Test = o_UpDown[int(len(i_Features)*1-200):len(i_Features)]

RFEstimator = rf(n_estimators=80, max_depth=2, random_state=0)
print(np.where(np.isnan(i_Train))[0])
print(np.where(np.isnan(o_Train))[0])
i_Train = np.delete(i_Train, np.where(np.isnan(i_Train))[0], axis=0)
o_Train = np.delete(o_Train, np.where(np.isnan(i_Train))[0], axis=0)
RFEstimator.fit(i_Train, o_Train)

print(RFEstimator.feature_importances_)

esti_trend = RFEstimator.predict(i_Test)
print(esti_trend)

np.save("600839_Estimation_Trend.npy", esti_trend)
np.save("600839_True_Trend.npy", o_Test)

TP=0
FP=0
TN=0
FN=0
for i in range(len(o_Test)):
    if esti_trend[i]==1:
        if esti_trend[i] == o_Test[i]:
            TP=TP+1
        else:
            FP=FP+1
    if esti_trend[i]==0:
        if esti_trend[i] == o_Test[i]:
            TN=TN+1
        else:
            FN=FN+1

acc=float((TP+TN) / (TP+TN+FP+FN))
print(acc)

