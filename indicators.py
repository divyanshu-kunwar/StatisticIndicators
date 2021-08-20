# import all the required files that is numpy , pandas and math library
import numpy as np
import pandas as pd
from pandas import DataFrame , Series
import math


# All the indicators are defined and arranged in Alphabetical order

# ---------------------> D <------------------------

# Double Exponential Moving Average (DEMA)
# 2 * EWMA - ewm(EWMA)
def dema(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    DEMA = (
    2*ema(data,period) - ema(data,period).ewm(span=period , adjust=adjust).mean()
    )
    return pd.Series(
        DEMA ,
        name = f'{period}_DEMA'
    )


# ---------------------> E <------------------------

# Exponential Weighted Moving Average (EWMA) or Exponential Moving Average(EMA)
# Exponential average of prev n day prices
def ema(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    return pd.Series(
        data[column].ewm(span=period, adjust=adjust).mean(),
        name = f'{period}_EMA'
    )

# Kaufman Efficiency indicator (KER) or (ER)
# change in price / volatility Here change and volatility are absolute
def er(data,period: int = 10,column: str ='close') -> Series:
    change = data[column].diff(period).abs()
    volatility = data[column].diff().abs().rolling(window=period,min_periods=1).sum()
    return pd.Series(change / volatility, 
        name=f'{period}_ER'
    )

# Elastic Volume Weighted Moving Average (EVWMA)
# x is ((volume sum for n period) - volume ) divided by (volume sum for n period)
# y is volume * close / (volume sum for n period)
def evwma(data, period: int = 20) -> Series:
    vol_sum = (data["volume"].rolling(window=period,min_periods=1).sum())

    x = (vol_sum - data["volume"]) / vol_sum
    y = (data["volume"] * data["close"]) / vol_sum
    
    evwma = [0]
    
    for x, y in zip(x.fillna(0).iteritems(), y.iteritems()):
            if x[1] == 0 or y[1] == 0:
                evwma.append(0)
            else:
                evwma.append(evwma[-1] * x[1] + y[1])

    return pd.Series(
        evwma[1:], index=data.index, 
        name=f'{period}_EVWMA'
    )


# ---------------------> F <------------------------

# Fractal Adaptive Moving Average (FRAMA)
# TODO 
def FRAMA(data: DataFrame, period: int = 16, batch: int=10) -> Series:

        assert period % 2 == 0, print("FRAMA period must be even")

        c = data.close.copy()
        window = batch * 2

        hh = c.rolling(batch).max()
        ll = c.rolling(batch).min()

        n1 = (hh - ll) / batch
        n2 = n1.shift(batch)

        hh2 = c.rolling(window).max()
        ll2 = c.rolling(window).min()
        n3 = (hh2 - ll2) / window

        # calculate fractal dimension
        D = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        alp = np.exp(-4.6 * (D - 1))
        alp = np.clip(alp, .01, 1).values

        filt = c.values
        for i, x in enumerate(alp):
            cl = c.values[i]
            if i < window:
                continue
            filt[i] = cl * x + (1 - x) * filt[i - 1]

        return pd.Series(filt, index=data.index, 
        name= f'{period} FRAMA'
        )


# ---------------------> H <------------------------

# Hull Moving Average (HMA)
# wma of change in wma where change in wma is 2 * (wma half period) - (wma full period) 
def hma(data, period: int = 16) -> Series:
    
    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wmaf = wma(data, period=half_length)
    wmas = wma(data, period=period)
    data["deltawma"] = 2 * wmaf - wmas
    hma = wma(data, column="deltawma", period=sqrt_length)

    return pd.Series(hma, name=f'{period}_HMA')


# ---------------------> K <------------------------

# Kaufman's Adaptive Moving Average (KAMA)
# first KAMA is SMA
# Current KAMA = Previous KAMA + smoothing_constant * (Price - Previous KAMA)

def kama(data,er_: int = 10,ema_fast: int = 2,
         ema_slow: int = 30,period: int = 20,
         column: str ='close') -> Series:
    er_ = er(data)
    fast_alpha = 2 / (ema_fast + 1)
    slow_alpha = 2 / (ema_slow + 1)
    sc = pd.Series(
            (er_ * (fast_alpha - slow_alpha) + slow_alpha) ** 2,
            name="smoothing_constant",
        )
    sma = pd.Series(
            data[column].rolling(period).mean(), name="SMA"
        )
    kama = []
    for s, ma, price in zip(
            sc.iteritems(), sma.shift().iteritems(), data[column].iteritems()
        ):
            try:
                kama.append(kama[-1] + s[1] * (price[1] - kama[-1]))
            except (IndexError, TypeError):
                if pd.notnull(ma[1]):
                    kama.append(ma[1] + s[1] * (price[1] - ma[1]))
                else:
                    kama.append(None)
    sma["KAMA"] = pd.Series(
            kama, index=sma.index,  name=f'{period}_KAMA')

    return sma['KAMA']


# ---------------------> M <------------------------

# Moving average convergence divergence (MACD)
# TODO
def macd(data,period_fast: int = 12,period_slow: int = 26,
        signal: int = 9,column: str = "close",adjust: bool = True
    ) -> DataFrame:
    
    EMA_fast = pd.Series(
            data[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name=f'{period_fast}_EMA_fast')
    EMA_slow = pd.Series(
        data[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
        name=f'{period_slow}_EMA_slow')
    MACD = pd.Series(EMA_fast - EMA_slow,name='MACD')
    MACD_signal = pd.Series(
        MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(),name=f'{signal}_SIGNAL'
    )
    DIFF = pd.Series(
        MACD - MACD_signal,
        name="diff MACD_MSIGNAL"
    )
    return pd.concat(
        [DIFF, MACD, MACD_signal ],
        axis=1
    )

# Moving Volume Weighted Average Price (MVWAP)
# SMA of (close * volume ) divided by SMA of volume
def mvwap(data: DataFrame, period:int = 9) -> Series:
        data["cv"] =(data["close"] * data["volume"])
        return pd.Series(
            (sma(data,period = period,column = "cv")/sma(data,period=period,column="volume")),
            name="MVWAP."
        )


# ---------------------> S <------------------------

# Simple moving average (SMA) or moving average (MA)
# Average of prev n day prices
def sma(data,period: int = 10,column: str ='close') -> Series:
    return pd.Series(
        data[column].rolling(window = period,min_periods= 1).mean(),
        name = f'{period}_SMA'
    )

# Simple moving median (SMM) or moving median (MM)
# median of prev n day prices
def smm(data,period: int = 10,column: str ='close') -> Series:
    return pd.Series(
        data[column].rolling(window = period,min_periods= 1).median(),
        name = f'{period}_SMM'
    )

# Simple smoothed moving average (SSMA) or smoothed moving average()
# smoothed (exponential + simple) average of prev n day prices
def ssma(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    return pd.Series(
        data[column].ewm(ignore_na = False, alpha=1.0/period, 
        min_periods=0, adjust=adjust).mean(),
        name = f'{period}_SSMA'
    )


# ---------------------> T <------------------------

# Triple Exponential Moving Average (TEMA)
# 3 * EWMA - ewm(ewm(ewm(data))) i.e. 3 * ewma - ewm of ewm of ewm of data

def tema(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    triple_ema = 3 * ema(data,period)
    ema_ema_ema = (
        ema(data,period).ewm(ignore_na = False, span = period, adjust = adjust).mean()
        .ewm(ignore_na = False, span = period, adjust = adjust).mean()
    )
    TEMA = (
    triple_ema - 3 * ema(data,period).ewm(span=period, adjust= adjust).mean() + ema_ema_ema
    )
    return pd.Series(
        TEMA ,
        name = f'{period}_TEMA'
    )

# Typical Price (TP)
# average of high low close price
def tp(data: DataFrame) -> Series:
        return pd.Series(
            (data["high"] + data["low"] + data["close"]) / 3,
             name="TP"
        )

# Triangular Moving Average (TRIMA) or (TMA)
# sum of SMA / period
def trima(data,period: int = 10,adjust: bool = True) -> Series:
    SMA = sma(data,period).rolling(window=period , min_periods=1).sum()
    return pd.Series(
        SMA / period,
        name = f'{period}_TRIMA'
    )

# Triple Exponential Average (TRIX)
# 1000*(m - mprev) / m Here m = ema(ema(ema(data))) or m = ema of ema of ema of data 

def trix(data,period: int = 10,adjust: bool = True,column: str ='close') -> Series:
    data_ = data[column]
    def _ema(data_, period, adjust):
        return pd.Series(data_.ewm(span=period, adjust=adjust).mean())

    m = _ema(_ema(_ema(data_, period, adjust), period, adjust), period, adjust)
    return pd.Series(
        10000 * (m.diff() / m), 
        name = f'{period}_TRIX'
    )

# ---------------------> V <------------------------

# Volume Adjusted Moving Average (VAMA)
# volume ratio = (price * volume) / mean of (price * volume) for n period
# cummulative sum = sum of (volume ratio * data) for n period
# cummulative Division = sum of (volume ratio) for n period
# VAMA = cummulative sum / cummulative Division

def vama(data,period: int = 10,column: str ='close') -> Series:
    vp = data[column]*data['volume']
    volsum = data["volume"].rolling(window=period,min_periods=1).mean()
    volRatio = pd.Series(vp / volsum, name="VAMA")
    cumSum = (volRatio * data[column]).rolling(window=period,min_periods=1).sum()
    cumDiv = volRatio.rolling(window=period,min_periods=1).sum()
    return pd.Series(
        cumSum / cumDiv, 
        name=f'{period}_VAMA'
    )

# Volume Weighted Average Price (VWAP)
# cummulative sum of (data) divided by cummulative sum of volume
def vwap(data: DataFrame) -> Series:
        return pd.Series(
            ((data["volume"] * tp(data)).cumsum()) / data["volume"].cumsum(),
            name="VWAP",
        )


# ---------------------> W <------------------------

# Weighted Moving Average (WMA)
# add weight to moving average
def wma(data, period: int = 9, 
        column: str = "close") -> Series:
    d = (period * (period + 1))/2
    weights = np.arange(1, period + 1)
    
    def linear(w):
            def _compute(x):
                return (w * x).sum() / d

            return _compute

    _close = data[column].rolling(period, min_periods=period)
    wma = _close.apply(linear(weights), raw=True)
    return pd.Series(
        wma, 
        name=f'{period}_WMA'
    )


# ---------------------> Z <------------------------

# Zero Lag Exponential Moving Average (ZLEMA)
# ema is sum of data and difference of data and data_lag
# ZLEMA is ewm of ema calculated
def zlema(data,period: int = 26, adjust: bool = True,
       column: str = "close") -> Series:
    lag = (period - 1) / 2

    ema = pd.Series(
            (data[column] + (data[column].diff(lag))),
            name=f'{period}_ZLEMA')
    zlema = pd.Series(
            ema.ewm(span=period, adjust=adjust).mean(),
            name=f'{period}_ZLEMA'
        )
    return zlema
