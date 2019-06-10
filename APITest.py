import tushare as ts
import pandas as pd
import numpy as np


def sign(x):
    if x <= 0:
        return 0
    elif x > 0:
        return 1


def ema(C, span=10):    #backward ema
    t_ema=0
    for i in range(len(C)):
        t_ema=t_ema+2/(span+1)*(C[len(C)-i-1]-t_ema)
    return t_ema

print(ts.__version__)
print(pd.__file__)

STK = ts.get_hist_data('600839', start='2015-01-05')

print(STK)

_n = 10
o_UpDown=[]
i_Features=np.ndarray((len(STK)-_n-1, 10), float)
# i_Features=[]

_EMA12=0
_EMA26=0
_MACD=0

for i in range(1,len(STK)-_n):
    o_UpDown.append((sign(STK.iloc[i-1]['p_change'])))

    _SMA = STK.iloc[i]['ma10']  #SMA

    _WMA = 0.0
    _nAA = 0
    _C=STK.iloc[i:i+_n]['close']
    _WMA=sum(_C.to_numpy()*np.linspace(10, 1, 10))/sum(np.arange(1,11))

    # _EMA =

    _MOM = (_C.head(1).to_numpy() - _C.tail(1).to_numpy())[0]    #MOM

    _H = STK.iloc[i:i+_n]['high']
    _L = STK.iloc[i:i+_n]['low']
    _HH = _H.sort_values(ascending=False)[0]
    _LL = _L.sort_values(ascending=True)[0]
    _STCK = ((_C.head(1)-_LL)/(_HH-_LL)*100).to_numpy()[0]   #STCK
    i_Features[i-1, 3]=_STCK

    # print(_STCK)

    if i<_n:
        _STCD = np.average(i_Features[0:i, 3])
    else:
        _STCD = np.average(i_Features[i-_n:i, 3])    #STCD


    # _EMA12 = _C.ewm(span=12).mean()[-1]
    # _EMA26 = _C.ewm(span=26).mean()[-1]

    _EMA12 = ema(_C, span=12)
    _EMA26 = ema(_C, span=26)

    _DIFF = _EMA12 - _EMA26
    _MACD = _MACD + 2 / (_n + 1) * (_DIFF - _MACD)  # MACD

    _priceChanges = STK.iloc[i:i+_n]['price_change']
    _UP, _DW = _priceChanges.copy(), _priceChanges.copy()
    _UP[_UP<0]=0
    _DW[_DW>0]=0
    # _UPSUM = _UP.ewm(span=12).mean()[-1]
    # _DWSUM = _DW.ewm(span=12).mean()[-1]
    _UPSUM = ema(_UP, span=12)
    _DWSUM = abs(ema(_DW, span=12))
    if _DWSUM == 0:
        _RSI=100
    else:
        _RSI = 100 - 100/(1+_UPSUM/_DWSUM)   #RSI

    _WILLR = ((_HH - _C.head(1))/(_HH - _LL) * 100).to_numpy()[0]    #WILLR

    _ADO = ((2*_C.head(1) - _H.head(1) - _L.head(1)) / (_H.head(1) - _L.head(1))).to_numpy()[0]  #ADO

    _M = (_H.to_numpy()+_L.to_numpy()+_C.to_numpy())/3
    _SM = sum(_M)/_n
    _D = sum(abs(_M-_SM))/_n
    _CCI = (_M[0] - _SM) / (0.015*_D)

    i_Features[i-1, 0]=_SMA
    i_Features[i-1, 1]=_WMA
    i_Features[i-1, 2]=_MOM
    i_Features[i-1, 4]=_STCD
    i_Features[i-1, 5]=_MACD
    i_Features[i-1, 6]=_RSI
    i_Features[i-1, 7]=_WILLR
    i_Features[i-1, 8]=_ADO
    i_Features[i-1, 9]=_CCI


print(o_UpDown)
print(i_Features)
print(len(o_UpDown))

np.save("600839_i.npy", i_Features)
np.save("600839_o.npy", o_UpDown)