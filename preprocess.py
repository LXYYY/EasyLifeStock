import pandas as pd
import numpy as np


def sign(x):
    if x <= 0:
        return 0
    elif x > 0:
        return 1


def ema(C, span=10):  # backward ema
    t_ema = 0
    for i in range(len(C)):
        t_ema = t_ema + 2 / (span + 1) * (C[len(C) - i - 1] - t_ema)
    return t_ema


class DataPrep:

    def __init__(self, apiType, STK, _n=10):
        if apiType=="quandl":
            STK=STK.reindex(index=STK.index[::-1])

        self._n = _n
        self.apiType = apiType
        self.STK = STK
        self.cols = STK.columns.values
        self.o_UpDown = []
        self.i_Features = np.ndarray((len(self.STK) - _n - 1, 10), float)
        print("api type: " + self.apiType)
        print("stk.size: " + str(len(self.STK)))

        if len(np.where(self.cols == 'close')[0]) != 0:
            self.closeCol = 'close'
        elif len(np.where(self.cols == 'Close')[0]) != 0:
            self.closeCol = 'Close'
        else:
            print("error: no close price !!!")
        print(self.closeCol)

        if len(np.where(self.cols == 'high')[0]) != 0:
            self.highCol = 'high'
        elif len(np.where(self.cols == 'High')[0]) != 0:
            self.highCol = 'High'
        else:
            print("error: no high price !!!")
        print(self.highCol)

        if len(np.where(self.cols == 'low')[0]) != 0:
            self.lowCol = 'low'
        elif len(np.where(self.cols == 'Low')[0]) != 0:
            self.lowCol = 'Low'
        else:
            print("error: no low price !!!")
        print(self.lowCol)

        if len(np.where(self.cols == 'open')[0]) != 0:
            self.openCol = 'low'
        elif len(np.where(self.cols == 'Open')[0]) != 0:
            self.openCol = 'Open'
        else:
            print("error: no open price !!!")
        print(self.openCol)

        if len(np.where(self.cols == 'volume')[0]) != 0:
            self.volumeCol = 'volume'
        elif len(np.where(self.cols == 'Volume')[0]) != 0:
            self.volumeCol = 'Volume'
        else:
            print("error: no volume !!!")
        print(self.volumeCol)

        if len(np.where(self.cols == 'price_change')[0]) != 0:
            self.priceChangeCol = 'price_change'
        else:
            price_change=np.zeros([len(self.STK),1])
            for i in range(0, len(self.STK) - 1):
                price_change[i] = self.STK.iloc[i][self.closeCol] - self.STK.iloc[i + 1][self.closeCol]
            self.STK.insert(1, 'price_change', price_change)
            self.priceChangeCol = 'price_change'
            print("warning: no price_change data, use cal data !!!")
        print(self.priceChangeCol)

        if len(np.where(self.cols == 'ma10')[0]) != 0:
            self.ma10Col = 'ma10'
        else:
            ma10=np.zeros([len(self.STK),1])
            for i in range(0, len(self.STK) - 10):
                _C = self.STK.iloc[i:i + 10][self.closeCol]
                ma10[i] = sum(_C.to_numpy()) / 10
            self.STK.insert(1, 'ma10', ma10)
            self.ma10Col = 'ma10'
            print("warning: no ma10 data, use cal data !!!")
        print(self.ma10Col)

    # def get_ma(self, row, win):
    #     if len(np.where(self.cols=='ma10')[0])==0:
    #         return self.STK.iloc[row]['ma10']
    #     else:
    #         _C=self.STK.iloc[row:row+win]['ma10']
    #         return sum(_C.to_numpy())/win

    def stk2ind(self):
        _n = self._n

        _EMA12 = 0
        _EMA26 = 0
        _MACD = 0

        for i in range(1, len(self.STK) - _n):
            self.o_UpDown.append((sign(self.STK.iloc[i - 1]['p_change'])))

            _SMA = self.STK.iloc[i][self.ma10Col]  # SMA

            _WMA = 0.0
            _nAA = 0
            _C = self.STK.iloc[i:i + _n][self.closeCol]
            _WMA = sum(_C.to_numpy() * np.linspace(10, 1, 10)) / sum(np.arange(1, 11))

            # _EMA =

            _MOM = (_C.head(1).to_numpy() - _C.tail(1).to_numpy())[0]  # MOM

            _H = self.STK.iloc[i:i + _n][self.highCol]
            _L = self.STK.iloc[i:i + _n][self.lowCol]
            _HH = _H.sort_values(ascending=False)[0]
            _LL = _L.sort_values(ascending=True)[0]
            _STCK = ((_C.head(1) - _LL) / (_HH - _LL) * 100).to_numpy()[0]  # STCK
            self.i_Features[i - 1, 3] = _STCK

            # print(_STCK)

            if i < _n:
                _STCD = np.average(self.i_Features[0:i, 3])
            else:
                _STCD = np.average(self.i_Features[i - _n:i, 3])  # STCD

            # _EMA12 = _C.ewm(span=12).mean()[-1]
            # _EMA26 = _C.ewm(span=26).mean()[-1]

            _EMA12 = ema(_C, span=12)
            _EMA26 = ema(_C, span=26)

            _DIFF = _EMA12 - _EMA26
            _MACD = _MACD + 2 / (_n + 1) * (_DIFF - _MACD)  # MACD

            _priceChanges = self.STK.iloc[i:i + _n][self.priceChangeCol]
            _UP, _DW = _priceChanges.copy(), _priceChanges.copy()
            _UP[_UP < 0] = 0
            _DW[_DW > 0] = 0
            # _UPSUM = _UP.ewm(span=12).mean()[-1]
            # _DWSUM = _DW.ewm(span=12).mean()[-1]
            _UPSUM = ema(_UP, span=12)
            _DWSUM = abs(ema(_DW, span=12))
            if _DWSUM == 0:
                _RSI = 100
            else:
                _RSI = 100 - 100 / (1 + _UPSUM / _DWSUM)  # RSI

            _WILLR = ((_HH - _C.head(1)) / (_HH - _LL) * 100).to_numpy()[0]  # WILLR

            _ADO = ((2 * _C.head(1) - _H.head(1) - _L.head(1)) / (_H.head(1) - _L.head(1))).to_numpy()[0]  # ADO

            _M = (_H.to_numpy() + _L.to_numpy() + _C.to_numpy()) / 3
            _SM = sum(_M) / _n
            _D = sum(abs(_M - _SM)) / _n
            _CCI = (_M[0] - _SM) / (0.015 * _D)

            self.i_Features[i - 1, 0] = _SMA
            self.i_Features[i - 1, 1] = _WMA
            self.i_Features[i - 1, 2] = _MOM
            self.i_Features[i - 1, 4] = _STCD
            self.i_Features[i - 1, 5] = _MACD
            self.i_Features[i - 1, 6] = _RSI
            self.i_Features[i - 1, 7] = _WILLR
            self.i_Features[i - 1, 8] = _ADO
            self.i_Features[i - 1, 9] = _CCI

        print(str(len(self.o_UpDown)) + " output labels prepared")
        print(str(len(self.i_Features)) + " input prepared")
        print(str(len(self.i_Features[0])) + " features prepared")

        return self.i_Features, self.o_UpDown

    def get_ind(self):
        return self.i_Features, self.o_UpDown
