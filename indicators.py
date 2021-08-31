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

# [0] __ Adaptive Price Zone (APZ)
# TODO
def apz(data: DataFrame,period: int = 21,dev_factor: int = 2,
    MA: Series = None,adjust: bool = True,) -> DataFrame:
    if not isinstance(MA, pd.Series):
        MA = dema(data, period)
    price_range = pd.Series(
        (data["high"] - data["low"]).ewm(span=period, adjust=adjust).mean()
    )
    volatility_value = pd.Series(
        price_range.ewm(span=period, adjust=adjust).mean(), name="vol_val"
    )

    upper_band = pd.Series((volatility_value * dev_factor) + MA, name="UPPER")
    lower_band = pd.Series(MA - (volatility_value * dev_factor), name="LOWER")

    return pd.concat([upper_band, lower_band], axis=1)


# ------------------> B <------------------------

# [0] __ Bollinger Bands (BBANDS)
# TODO
def bbands(data: DataFrame,period: int = 20,MA: Series = None,
        column: str = "close",std_multiplier: float = 2,) -> DataFrame:

        std = data[column].rolling(window=period).std()

        if not isinstance(MA, pd.core.series.Series):
            middle_band = pd.Series(sma(data, period), name="BB_MIDDLE")
        else:
            middle_band = pd.Series(MA, name="BB_MIDDLE")

        upper_bb = pd.Series(middle_band + (std_multiplier * std), name="BB_UPPER")
        lower_bb = pd.Series(middle_band - (std_multiplier * std), name="BB_LOWER")

        return pd.concat([upper_bb, middle_band, lower_bb], axis=1)

# [0] __ Bollinger Bands Width (BBWidth)
# TODO
def bbwidth(
         data: DataFrame, period: int = 20, MA: Series = None, column: str = "close"
    ) -> Series:
        

        BB = bbands(data, period, MA, column)

        return pd.Series(
            (BB["BB_UPPER"] - BB["BB_LOWER"]) / BB["BB_MIDDLE"],
            name="{0} period BBWITH".format(period),
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

# [0] __ Directional Movement Index (DMI)
# TODO
def dmi(data: DataFrame, column: str = "close", adjust: bool = True) -> Series:
    def _get_time(close):
        sd = close.rolling(5).std()
        asd = sd.rolling(10).mean()
        v = sd / asd
        t = 14 / v.round()
        t[t.isna()] = 0
        t = t.map(lambda x: int(min(max(x, 5), 30)))
        return t
    def _dmi(index):
        time = t.iloc[index]
        if (index - time) < 0:
            subset = data.iloc[0:index]
        else:
            subset = data.iloc[(index - time) : index]
        return rsi(subset, period=time, adjust=adjust).values[-1]
    dates = Series(data.index)
    periods = Series(range(14, len(dates)), index=dates.index[14:].values)
    t = _get_time(data[column])
    return periods.map(lambda x: _dmi(x))

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

# [0] __ TODO (EVSTC)
# TODO
def evstc(data: DataFrame,period_fast: int = 12,period_slow: int = 30,
        k_period: int = 10,d_period: int = 3,adjust: bool = True) -> Series:
        
        ema_slow = evwma(data, period_slow)
        ema_fast = evwma(data, period_fast)

        macd = ema_fast - ema_slow

        STOK = pd.Series((
            (macd - macd.rolling(window=k_period).min())
            / (macd.rolling(window=k_period).max() - macd.rolling(window=k_period).min())
            ) * 100)

        STOD = STOK.rolling(window=d_period).mean()
        STOD_DoubleSmooth = STOD.rolling(window=d_period).mean()

        return pd.Series(STOD_DoubleSmooth, name="{0} period EVSTC".format(k_period))

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

# [0] __  Fisher Transform
# TODO
def fish(data: DataFrame, period: int = 10, adjust: bool = True) -> Series:
    from numpy import log, seterr

    seterr(divide="ignore")

    med = (data["high"] + data["low"]) / 2
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    raw = (2 * ((med - ndaylow) / (ndayhigh - ndaylow))) - 1
    smooth = raw.ewm(span=5, adjust=adjust).mean()
    _smooth = smooth.fillna(0)

    return pd.Series(
        (log((1 + _smooth) / (1 - _smooth))).ewm(span=3, adjust=adjust).mean(),
        name="{0} period FISH.".format(period),
    )

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

# [0] __ Finite Volume Element (FVE)
# TODO
def fve(data: DataFrame, period: int = 22, factor: int = 0.3) -> Series:
        hl2 = (data["high"] + data["low"]) / 2
        tp_ = tp(data)
        smav = data["volume"].rolling(window=period).mean()
        mf = pd.Series((data["close"] - hl2 + tp_.diff()), name="mf")
        _mf = pd.concat([data["close"], data["volume"], mf], axis=1)

        def vol_shift(row):

            if row["mf"] > factor * row["close"] / 100:
                return row["volume"]
            elif row["mf"] < -factor * row["close"] / 100:
                return -row["volume"]
            else:
                return 0

        _mf["vol_shift"] = _mf.apply(vol_shift, axis=1)
        _sum = _mf["vol_shift"].rolling(window=period).sum()

        return pd.Series((_sum / smav) / period * 100)


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

# ------------------> I <------------------------

# [0] __ Ichimoku Cloud
# TODO
def ichimoku(data: DataFrame,tenkan_period: int = 9,kijun_period: int = 26,
    senkou_period: int = 52,chikou_period: int = 26,) -> DataFrame:
    tenkan_sen = pd.Series(
        (
            data["high"].rolling(window=tenkan_period).max()
            + data["low"].rolling(window=tenkan_period).min()
        )
        / 2,
        name="TENKAN",
    )  ## conversion line

    kijun_sen = pd.Series(
        (
            data["high"].rolling(window=kijun_period).max()
            + data["low"].rolling(window=kijun_period).min()
        )
        / 2,
        name="KIJUN",
    )  ## base line

    senkou_span_a = pd.Series(
        ((tenkan_sen + kijun_sen) / 2), name="senkou_span_a"
    ) .shift(kijun_period) ## Leading span

    senkou_span_b = pd.Series(
        (
            (
                data["high"].rolling(window=senkou_period).max()
                + data["low"].rolling(window=senkou_period).min()
            )
            / 2
        ),
        name="SENKOU",
    ).shift(kijun_period)

    chikou_span = pd.Series(
        data["close"].shift(-chikou_period),
        name="CHIKOU",
    )

    return pd.concat(
        [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1
    )

# [0] __ Inverse Fisher Transform (IFTRSI)
# TODO
def ift_rsi(data: DataFrame,column: str = "close",rsi_period: int = 5,
           wma_period: int = 9,) -> Series:
    v1 = pd.Series(0.1 * (rsi(data, rsi_period) - 50), name="v1")
    d = (wma_period * (wma_period + 1)) / 2 
    weights = np.arange(1, wma_period + 1)

    def linear(w):
        def _compute(x):
            return (w * x).sum() / d

        return _compute

    _wma = v1.rolling(wma_period, min_periods=wma_period)
    v2 = _wma.apply(linear(weights), raw=True)

    return pd.Series(
        ((v2 ** 2 - 1) / (v2 ** 2 + 1)), 
        name="IFT_RSI"
    )


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

# [0] __ Keltner Channels (KC)
# TODO
def kc(ohlc: DataFrame,period: int = 20,atr_period: int = 10,
       MA: Series = None,kc_mult: float = 2,) -> DataFrame:

        if not isinstance(MA, pd.core.series.Series):
            middle = pd.Series(ema(ohlc, period), name="KC_MIDDLE")
        else:
            middle = pd.Series(MA, name="KC_MIDDLE")

        up = pd.Series(middle + (kc_mult * atr(ohlc, atr_period)), name="KC_UPPER")
        down = pd.Series(
            middle - (kc_mult * atr(ohlc, atr_period)), name="KC_LOWER"
        )

        return pd.concat([up, down], axis=1)
  

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

# [0] __ Moving Standard Deviation (MSD)
# Standard deviation of a given period for the column passed as arguement
def msd(data: DataFrame, period: int = 21, column: str = "close") -> Series:
        return pd.Series(data[column].rolling(period).std(), name="MSD")

# Momentum Breakout Bands (MOBO)
# TODO
def mobo(data: DataFrame,period: int = 10,std_multiplier: float = 0.8,
         column: str = "close",) -> DataFrame:

        BB = bbands(data, period=10, std_multiplier=0.8, column=column)
        return BB

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

# [0] __ Stop And Reverse (SAR) 
# The indicator is below prices when prices are rising and above 
# prices when prices are falling.
# TODO
def sar(data: DataFrame, af: int = 0.02, amax: int = 0.2) -> Series:
        
        high, low = data.high, data.low
        # Starting values
        sig0, xpt0, af0 = True, high[0], af
        _sar = [low[0] - (high - low).std()]

        for i in range(1, len(data)):
            sig1, xpt1, af1 = sig0, xpt0, af0

            lmin = min(low[i - 1], low[i])
            lmax = max(high[i - 1], high[i])

            if sig1:
                sig0 = low[i] > _sar[-1]
                xpt0 = max(lmax, xpt1)
            else:
                sig0 = high[i] >= _sar[-1]
                xpt0 = min(lmin, xpt1)

            if sig0 == sig1:
                sari = _sar[-1] + (xpt1 - _sar[-1]) * af1
                af0 = min(amax, af1 + af)

                if sig0:
                    af0 = af0 if xpt0 > xpt1 else af1
                    sari = min(sari, lmin)
                else:
                    af0 = af0 if xpt0 < xpt1 else af1
                    sari = max(sari, lmax)
            else:
                af0 = af
                sari = xpt0

            _sar.append(sari)

        return pd.Series(_sar, index=data.index)

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

# [0] __ The Schaff Trend Cycle (Oscillator) (STC)
# TODO
def stc(data: DataFrame,period_fast: int = 23,period_slow: int = 50,k_period: int = 10,
        d_period: int = 3,column: str = "close",adjust: bool = True) -> Series:
        EMA_fast = pd.Series(
            data[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )

        EMA_slow = pd.Series(
            data[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )

        MACD = pd.Series((EMA_fast - EMA_slow), name="MACD")

        STOK = pd.Series((
            (MACD - MACD.rolling(window=k_period).min())
            / (MACD.rolling(window=k_period).max() - MACD.rolling(window=k_period).min())
            ) * 100)

        STOD = STOK.rolling(window=d_period).mean()
        STOD_DoubleSmooth = STOD.rolling(window=d_period).mean()  # "double smoothed"
        return pd.Series(STOD_DoubleSmooth, name="{0} period STC".format(k_period))

# [0] __ (SQZMI)
# TODO
def sqzmi(data: DataFrame, period: int = 20, MA: Series = None) -> DataFrame:

    if not isinstance(MA, pd.core.series.Series):
        ma = pd.Series(sma(data, period))
    else:
        ma = None

    bb = bbands(data, period=period, MA=ma)
    kc_ = kc(data, period=period, kc_mult=1.5)
    comb = pd.concat([bb, kc_], axis=1)

    def sqz_on(row):
        if row["BB_LOWER"] > row["KC_LOWER"] and row["BB_UPPER"] < row["KC_UPPER"]:
            return True
        else:
            return False

    comb["SQZ"] = comb.apply(sqz_on, axis=1)

    return pd.Series(comb["SQZ"], name="{0} period SQZMI".format(period))


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

# [0] __ Volume Price Trend (VPT)
# TODO
def vpt(data: DataFrame) -> Series:
        hilow = (data["high"] - data["low"]) * 100
        openclose = (data["close"] - data["open"]) * 100
        vol = data["volume"] / hilow
        spreadvol = (openclose * vol).cumsum()

        vpt = spreadvol + spreadvol

        return pd.Series(vpt, name="VPT")

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

# ------------------> V <------------------------

# [0] __ Volume Flow Indicator (VFI)
# TODO
def vfi(data: DataFrame,period: int = 130,smoothing_factor: int = 3,factor: int = 0.2,
        vfactor: int = 2.5,adjust: bool = True,) -> Series:
        typical = tp(data)
        inter = typical.apply(np.log).diff()
        vinter = inter.rolling(window=30).std()
        cutoff = pd.Series(factor * vinter * data["close"], name="cutoff")
        price_change = pd.Series(typical.diff(), name="pc")  
        mav = pd.Series(
            data["volume"].rolling(center=False, window=period).mean(), name="mav",
        )

        _va = pd.concat([data["volume"], mav.shift()], axis=1)
        _mp = pd.concat([price_change, cutoff], axis=1)
        _mp.fillna(value=0, inplace=True)

        def _vol_added(row):
            if row["volume"] > vfactor * row["mav"]:
                return vfactor * row["mav"]
            else:
                return row["volume"]

        added_vol = _va.apply(_vol_added, axis=1)

        def _multiplier(row):
            if row["pc"] > row["cutoff"]:
                return 1
            elif row["pc"] < 0 - row["cutoff"]:
                return -1
            else:
                return 0

        multiplier = _mp.apply(_multiplier, axis=1)
        raw_sum = (multiplier * added_vol).rolling(window=period).sum()
        raw_value = raw_sum / mav.shift()

        vfi = pd.Series(
            raw_value.ewm(
                ignore_na=False,
                min_periods=smoothing_factor - 1,
                span=smoothing_factor,
                adjust=adjust,
            ).mean(),
            name="VFI",
        )

        return vfi

# [0] __ Value chart (VC)
# TODO
def vc(data: DataFrame, period: int = 5) -> DataFrame:
       
        float_axis = ((data.high + data.low) / 2).rolling(window=period).mean()
        vol_unit = (data.high - data.low).rolling(window=period).mean() * 0.2

        value_chart_high = pd.Series((data.high - float_axis) / vol_unit, name="Value Chart High")
        value_chart_low = pd.Series((data.low - float_axis) / vol_unit, name="Value Chart Low")
        value_chart_close = pd.Series((data.close - float_axis) / vol_unit, name="Value Chart Close")
        value_chart_open = pd.Series((data.open - float_axis) / vol_unit, name="Value Chart Open")

        return pd.concat([value_chart_high, value_chart_low, value_chart_close, value_chart_open], axis=1)

# ------------------> W <------------------------

# [0] __ williams Fractal
# TODO
def williams_fractal(data: DataFrame, period: int = 2) -> DataFrame:
        def is_bullish_fractal(x):
            if x[period] == min(x):
                return True
            return False

        def is_bearish_fractal(x):
            if x[period] == max(x):
                return True
            return False

        window_size = period * 2 + 1
        bearish_fractals = pd.Series(
            data.high.rolling(window=window_size, center=True).apply(
                is_bearish_fractal, raw=True
            ),
            name="BearishFractal",
        )
        bullish_fractals = pd.Series(
            data.low.rolling(window=window_size, center=True).apply(
                is_bullish_fractal, raw=True
            ),
            name="BullishFractal",
        )
        return pd.concat([bearish_fractals, bullish_fractals], axis=1)

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

# [0] __ Wave Trend Oscillator (WTO)
# TODO
def wto(data: DataFrame,channel_length: int = 10,average_length: int = 21,adjust: bool = True,) -> DataFrame:

    ap = tp(data)
    esa = ap.ewm(span=average_length, adjust=adjust).mean()
    d = pd.Series(
        (ap - esa).abs().ewm(span=channel_length, adjust=adjust).mean(), name="d"
    )
    ci = (ap - esa) / (0.015 * d)

    wt1 = pd.Series(ci.ewm(span=average_length, adjust=adjust).mean(), name="WT1.")
    wt2 = pd.Series(wt1.rolling(window=4).mean(), name="WT2.")

    return pd.concat([wt1, wt2], axis=1)

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

