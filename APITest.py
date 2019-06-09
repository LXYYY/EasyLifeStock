import tushare as ts
import pandas as pd
import numpy as np


def sign(x):
    if x <= 0:
        return -1
    elif x > 0:
        return 1


print(ts.__version__)

STK = ts.get_hist_data('600839')

print(STK)
print(sign(STK.loc['2016-12-12']['p_change']))

_n = 10
o_UpDown=[]
i_Features=np.ndarray((len(STK-_n), 10), float)
# i_Features=[]

_EMA12=0
_EMA26=0
_MACD=0

for i in range(_n,len(STK)):
    o_UpDown.append((sign(STK.iloc[i+1]['p_change'])))
    _C = np.ndarray((_n,),float)

    _SMA = STK.iloc[i]['ma10']  #SMA

    _WMA = 0.0
    _nAA = 0
    _C=STK.iloc[i-_n+1:i+1]['close']
    _WMA=sum(_C.to_numpy()*np.arange(1,11))/sum(np.arange(1,11))

    # _EMA =

    _MOM = _C.head(1) - _C.tail(1)    #MOM

    _H = STK.iloc[i-_n+1:i+1]['high']
    _L = STK.iloc[i-_n+1:i+1]['low']
    _HH = _H.sort_values(ascending=False)[0]
    _LL = _L.sort_values(ascending=True)[0]
    _STCK = (_C.tail(1)-_LL)/(_HH-_LL)*100   #STCK
    # print(_STCK)

    if i >= _n*2:
        _STCD = sum(i_Features[i-_n+1:i, 3])*10    #STCD
    else:
        _STCD=0

    _alpha12=2/(12+1)
    _alpha26=2/(26+1)
    if i == _n:
        _EMA12=0
        _EMA26=0
        _MACD=0
    else:
        _EMA12 = _EMA12 + _alpha12*(_C[0]-_EMA12)
        _EMA26 = _EMA26 + _alpha26*(_C[0]-_EMA26)
        _DIFF = _EMA12-_EMA26
        _MACD = _MACD + 2/(10+1)*(_DIFF-_MACD)      #MACD

    _priceChanges = STK.iloc[i-_n+1:i+1]['price_change']
    _UPSUM = sum(_priceChanges[_priceChanges>0])
    _DWSUM = -sum(_priceChanges[_priceChanges<=0])
    _RSI = 100 - 100/(1+(_UPSUM/_n)/(_DWSUM/_n))    #RSI

    _WILLR = (_HH - _C.tail(1))/(_HH - _LL) * 100    #WILLR

    _ADO = (2*_C.tail(1) - _H.tail(1) - _L.tail(1)) / (_H.tail(1) - _L.tail(1))  #ADO

    _M = (_H.to_numpy()+_L.to_numpy()+_C.to_numpy())/3
    _SM = sum(_M)/_n
    _D = sum(abs(_M-_SM))/_n
    _CCI = (_M.tail(1) - _SM) / (0.015*_D)

    _Feature=np.ndarray([_SMA, _WMA, _MOM, _STCK, _STCD, _MACD, _RSI, _WILLR, _ADO, _CCI])
    i_Features[i-_n]=_Feature

print(o_UpDown)

print(len(o_UpDown))
