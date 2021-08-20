# import all the required files i.e. numpy , pandas and math library
import numpy as np
import pandas as pd
from pandas import DataFrame , Series
import math


# All the indicators are defined and arranged in Alphabetical order

# ------------------> A <------------------------

# [0] __ Average True Range (ATR)
# Moving Average of True Range(TR)
def atr(data: DataFrame, period: int = 14) -> Series:
        TR = tr(data)
        return pd.Series(
            TR.rolling(center=False, window=period, 
                       min_periods=1).mean(),
            name=f'{period}  ATR'
        )


# ------------------> D <------------------------

# [0] __ Double Exponential Moving Average (DEMA)
# 2 * EWMA - ewm(EWMA)
def dema(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    DEMA = (
    2*ema(data,period) - ema(data,period).ewm(span=period , adjust=adjust).mean()
    )
    return pd.Series(
        DEMA ,
        name = f'{period}_DEMA'
    )


# ------------------> E <------------------------

# [0] __ Exponential Weighted Moving Average (EWMA) or Exponential Moving Average(EMA)
# Exponential average of prev n day prices
def ema(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    return pd.Series(
        data[column].ewm(span=period, adjust=adjust).mean(),
        name = f'{period}_EMA'
    )

# [0] __ Kaufman Efficiency indicator (KER) or (ER)
# change in price / volatility Here change and volatility are absolute
def er(data,period: int = 10,column: str ='close') -> Series:
    change = data[column].diff(period).abs()
    volatility = data[column].diff().abs().rolling(window=period,min_periods=1).sum()
    return pd.Series(change / volatility, 
        name=f'{period}_ER'
    )

# [0] __ Elastic Volume Weighted Moving Average (EVWMA)
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

# [0] __ Elastic Volume Weighted Moving average convergence divergence (EV_MACD)
# MACD calculation on basis of Elastic Volume Weighted Moving average (EVWMA)
def ev_macd(data: DataFrame,period_fast: int = 20,period_slow: int = 40,
            signal: int = 9,adjust: bool = True,) -> DataFrame:
       
        evwma_slow = evwma(data, period_slow)

        evwma_fast = evwma(data, period_fast)

        MACD = pd.Series(evwma_fast - evwma_slow, name="EV MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
        )

        return pd.concat([MACD, MACD_signal], axis=1)


# ------------------> F <------------------------

# [0] __ Fractal Adaptive Moving Average (FRAMA)
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


# ------------------> H <------------------------

# [0] __ Hull Moving Average (HMA)
# wma of change in wma where change in wma is 2 * (wma half period) - (wma full period) 
def hma(data, period: int = 16) -> Series:
    
    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wmaf = wma(data, period=half_length)
    wmas = wma(data, period=period)
    data["deltawma"] = 2 * wmaf - wmas
    hma = wma(data, column="deltawma", period=sqrt_length)

    return pd.Series(hma, name=f'{period}_HMA')


# ------------------> K <------------------------

# [0] __ Kaufman's Adaptive Moving Average (KAMA)
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


# ------------------> M <------------------------

# [0] __ Moving average convergence divergence (MACD)
# MACD is Difference of ema fast and ema slow
# Here fast period is 12 and slow period is 26
# MACD Signal is ewm of MACD
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

# [0] __ Market momentum (MOM)
def mom(data: DataFrame, period: int = 10, column: str = "close") -> Series:

        return pd.Series(data[column].diff(period), 
                         name=f'{period}_MOM'
                        )

# [0] __ Moving Volume Weighted Average Price (MVWAP)
# SMA of (close * volume ) divided by SMA of volume
def mvwap(data: DataFrame, period:int = 9) -> Series:
        data["cv"] =(data["close"] * data["volume"])
        return pd.Series(
            (sma(data,period = period,column = "cv")/sma(data,period=period,column="volume")),
            name="MVWAP."
        )

# ------------------> P <------------------------

# ------------|| Pivot ||------------------------

# [0] __ Pivot Camarilla
# TODO
def pivot_camarilla(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot = pd.Series(tp(df_), name="pivot")
    
    s1 =  df_['close']+(1.1*(df_['high']-df_['low'])/12)
    s2 = df_['close']-(1.1*(df_['high']-df_['low'])/6)
    s3 = df_['close']-(1.1*(df_['high']-df_['low'])/4)
    s4 =df_['close']-(1.1*(df_['high']-df_['low'])/2)
   
    

    r1 = df_['close']+(1.1*(df_['high']-df_['low'])/12)
    r2 = df_['close']+(1.1*(df_['high']-df_['low'])/6)
    r3 =df_['close']+(1.1*(df_['high']-df_['low'])/4)
    r4 = df_['close']+(1.1*(df_['high']-df_['low'])/2)
   
    return pd.concat(
            [
                pivot,
                pd.Series(s1, name="s1"),
                pd.Series(s2, name="s2"),
                pd.Series(s3, name="s3"),
                pd.Series(s4, name="s4"),
                pd.Series(r1, name="r1"),
                pd.Series(r2, name="r2"),
                pd.Series(r3, name="r3"),
                pd.Series(r4, name="r4"),
                            ],
            axis=1,
        )

# [0] __ Pivot Classic
# TODO
def pivot_classic(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot = pd.Series(tp(df_), name="pivot")
    
    s1 = (pivot * 2) - df_["high"]
    s2 = pivot - (df_["high"] - df_["low"])
    s3 = pivot - 2*(df_["high"] - df_["low"])
    s4 = pivot - 3*(df_["high"] - df_["low"])
    
    

    r1 = (pivot * 2) - df_["low"]
    r2 = pivot + (df_["high"] - df_["low"])
    r3 = pivot + 2*(df_["high"] - df_["low"])
    r4 = pivot + 3*(df_["high"] - df_["low"])
   
    return pd.concat(
            [
                pivot,
                pd.Series(s1, name="s1"),
                pd.Series(s2, name="s2"),
                pd.Series(s3, name="s3"),
                pd.Series(s4, name="s4"),
                pd.Series(r1, name="r1"),
                pd.Series(r2, name="r2"),
                pd.Series(r3, name="r3"),
                pd.Series(r4, name="r4"),
               
            ],
            axis=1,
        )

# [0] __ Pivot Demark
# TODO
def pivot_demark(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot,s1,r1=[],[],[]
    for i in range(len(df_)):
        if df_['open'][i]==df_['close'][i]:
            x=df_['high'][i]+df_['low'][i]+2*df_['close'][i]
        elif df_['close'][i]>df_['open'][i]:
            x=2*df_['high'][i]+df_['low'][i]+df_['close'][i]
        else:
            x=df_['high'][i]+2*df_['low'][i]+df_['close'][i]
   
        pivot.append(x/4)
        s1.append(x/2 - df_["high"][i])

        r1.append(x/2 - df_["low"][i])
    
    data_ = pd.DataFrame(pivot,columns=['pivot'])
    data_['s1']=s1
    data_['r1']=r1
    return data_

# [0] __ Pivot Fibonacci
# TODO
def pivot_fibonacci(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot = pd.Series(tp(df_), name="pivot")
    
    s1 = pivot - ((df_["high"] - df_["low"])*0.382)
    s2 = pivot - ((df_["high"] - df_["low"])*0.618)
    s3 = pivot - (df_["high"] - df_["low"])
    s4 = pivot + ((df_["high"] - df_["low"])*1.382)
   
    

    r1 = pivot + ((df_["high"] - df_["low"])*0.382)
    r2 = pivot + ((df_["high"] - df_["low"])*0.618)
    r3 =pivot + (df_["high"] - df_["low"])
    r4 = pivot + (df_["high"] - df_["low"])*1.382
   
    return pd.concat(
            [
                pivot,
                pd.Series(s1, name="s1"),
                pd.Series(s2, name="s2"),
                pd.Series(s3, name="s3"),
                pd.Series(s4, name="s4"),
                pd.Series(r1, name="r1"),
                pd.Series(r2, name="r2"),
                pd.Series(r3, name="r3"),
                pd.Series(r4, name="r4"),
                            ],
            axis=1,
        )

# [0] __ Pivot Traditional
# TODO
def pivot_traditional(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot = pd.Series(tp(df_), name="pivot")
    
    s1 = (pivot * 2) - df_["high"]
    s2 = pivot - (df_["high"] - df_["low"])
    s3 = df_["low"] - (2 * (df_["high"] - pivot))
    s4 = df_["low"] - (3 * (df_["high"] - pivot))
    s5 = df_["low"] - (4 * (df_["high"] - pivot))
    

    r1 = (pivot * 2) - df_["low"]
    r2 = pivot + (df_["high"] - df_["low"])
    r3 = df_["high"] + (2 * (pivot - df_["low"]))
    r4 = df_["high"] + (3 * (pivot - df_["low"]))
    r5 = df_["high"] + (4 * (pivot - df_["low"]))
    return pd.concat(
            [
                pivot,
                pd.Series(s1, name="s1"),
                pd.Series(s2, name="s2"),
                pd.Series(s3, name="s3"),
                pd.Series(s4, name="s4"),
                pd.Series(s5, name="s5"),
                pd.Series(r1, name="r1"),
                pd.Series(r2, name="r2"),
                pd.Series(r3, name="r3"),
                pd.Series(r4, name="r4"),
                pd.Series(r5, name="r5"),
            ],
            axis=1,
        )

# [0] __ Pivot Woodie
# TODO
def pivot_woodie(data: DataFrame) -> DataFrame:
    df_ = data.shift()
    pivot = pd.Series((df_['high']+df_['low']+2*data['open'])/4, name="pivot")
    
    s1 =  2*pivot-df_['high']
    s2 = pivot - (df_["high"] - df_["low"])
    s3 = df_["low"] - (2 * (pivot - df_["high"]))
    s4 =  s3 - (df_["high"] - df_["low"])
   
    

    r1 = 2*pivot-df_['low']
    r2 = pivot + (df_["high"] - df_["low"])
    r3 =df_["high"] + (2 * (pivot - df_["low"]))
    r4 =  r3 + (df_["high"] - df_["low"])
   
    return pd.concat(
            [
                pivot,
                pd.Series(s1, name="s1"),
                pd.Series(s2, name="s2"),
                pd.Series(s3, name="s3"),
                pd.Series(s4, name="s4"),
                pd.Series(r1, name="r1"),
                pd.Series(r2, name="r2"),
                pd.Series(r3, name="r3"),
                pd.Series(r4, name="r4"),
                            ],
            axis=1,
        )

# [0] __ PPO
# TODO
def ppo(data: DataFrame,period_fast: int = 12,period_slow: int = 26,
    signal: int = 9,column: str = "close",
      adjust: bool = True,) -> DataFrame:

    EMA_fast = pd.Series(
        data[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
        name="EMA_fast",
    )
    EMA_slow = pd.Series(
        data[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
        name="EMA_slow",
    )
    PPO = pd.Series(((EMA_fast - EMA_slow) / EMA_slow) * 100, name="PPO")
    PPO_signal = pd.Series(
        PPO.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
    )
    PPO_histo = pd.Series(PPO - PPO_signal, name="HISTO")

    return pd.concat([PPO, PPO_signal, PPO_histo], axis=1)

# ------------------> R <------------------------

# [0] __ Relative Strength Index (RSI)
# EMA of up and down gives gain and loss
# Relative Strength Index is gain / loss
def rsi(data: DataFrame, period: int = 14,column: str = "close",
    adjust: bool = True,) -> Series:
    delta = data[column].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
    _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), 
                     name=f'{period} period RSI'
                    )

# [0] __ Rate of Change (ROC)
def roc(data: DataFrame, period: int = 12, column: str = "close") -> Series:
    return pd.Series(
        (data[column].diff(period) / data[column].shift(period)) * 100, 
        name="ROC"
    )


# ------------------> S <------------------------

# [0] __ Simple moving average (SMA) or moving average (MA)
# Average of prev n day prices
def sma(data,period: int = 10,column: str ='close') -> Series:
    return pd.Series(
        data[column].rolling(window = period,min_periods= 1).mean(),
        name = f'{period}_SMA'
    )

# [0] __ Simple moving median (SMM) or moving median (MM)
# median of prev n day prices
def smm(data,period: int = 10,column: str ='close') -> Series:
    return pd.Series(
        data[column].rolling(window = period,min_periods= 1).median(),
        name = f'{period}_SMM'
    )

# [0] __ Simple smoothed moving average (SSMA) or smoothed moving average()
# smoothed (exponential + simple) average of prev n day prices
def ssma(data,period: int = 10,column: str ='close',adjust: bool = True) -> Series:
    return pd.Series(
        data[column].ewm(ignore_na = False, alpha=1.0/period, 
        min_periods=0, adjust=adjust).mean(),
        name = f'{period}_SSMA'
    )


# ------------------> T <------------------------

# [0] __ Triple Exponential Moving Average (TEMA)
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

# [0] __ Typical Price (TP)
# average of high low close price
def tp(data: DataFrame) -> Series:
        return pd.Series(
            (data["high"] + data["low"] + data["close"]) / 3,
             name="TP"
        )

# [0] __ True Range (TR)
# maximum of three price ranges i.e TR1, TR2, TR2
def tr(data: DataFrame) -> Series:
        TR1 = pd.Series(data["high"] - data["low"]).abs()
        TR2 = pd.Series(data["high"] - data["close"].shift()).abs()
        TR3 = pd.Series(data["close"].shift() - data["low"]).abs()
        _TR = pd.concat([TR1, TR2, TR3], axis=1)
        _TR["TR"] = _TR.max(axis=1)
        return pd.Series(_TR["TR"], 
                         name="TR"
                        )


# [0] __ Triangular Moving Average (TRIMA) or (TMA)
# sum of SMA / period
def trima(data,period: int = 10,adjust: bool = True) -> Series:
    SMA = sma(data,period).rolling(window=period , min_periods=1).sum()
    return pd.Series(
        SMA / period,
        name = f'{period}_TRIMA'
    )

# [0] __ Triple Exponential Average (TRIX)
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

# ------------------> V <------------------------

# [0] __ Volume Adjusted Moving Average (VAMA)
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

# [0] __ Volume Weighted Average Price (VWAP)
# cummulative sum of (data) divided by cummulative sum of volume
def vwap(data: DataFrame) -> Series:
        return pd.Series(
            ((data["volume"] * tp(data)).cumsum()) / data["volume"].cumsum(),
            name="VWAP",
        )

# [0] __ Volume Weighted Moving average convergence divergence(VWMACD)
# difference vwma of fast and slow
def vw_macd(data: DataFrame,period_fast: int = 12,period_slow: int = 26,
        signal: int = 9,column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:

    MACD = pd.Series(vwma(data,period=period_fast)-vwma(data,period=period_slow), 
                     name="VW MACD")
    print(MACD)
   
    MACD_signal = pd.Series(
        MACD.ewm(span=signal, adjust=adjust).mean(),
        name="MACD Signal"
    )

    return pd.concat([MACD, MACD_signal], axis=1)

# [0] __ Volume Weighted Moving Average (VWMA)
# sum of (data * volume) for n period divided by
# sum of volume for n period
def vwma(data: DataFrame,period: int = 20,column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:
    
    cv=(data[column]*data['volume']).rolling(window=period,min_periods=1).sum()
    v=data['volume'].rolling(window=period,min_periods=1).sum()
    
    return pd.Series(cv/v,name='VWMA')


# ------------------> W <------------------------

# [0] __ Weighted Moving Average (WMA)
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


# ------------------> Z <------------------------

# [0] __ Zero Lag Exponential Moving Average (ZLEMA)
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

