import numpy as np
import pandas as pd

import preprocess as prep
import predict as tr

STK=pd.read_csv("GOOGL.csv")
if len(STK)==0:
    print("no csv data")
    exit()

dataPrep=prep.DataPrep("quandl", STK)
features, labels=dataPrep.stk2ind()

RF=tr.Predictor(features, labels, "rf")
RF.test()

