# StatisticIndicator
 
Statistic Indicator is a collection of 50+ stock analysis indicators that are being continiousally updated and added.

## How to use ?

**Step 1 :** Download the files. keep the indicators.py file in your working directory ( i.e. the directory in which you are going to write your program ) </br>
**Step 2 :** import the file or the function you would like to use in your python file as :
>import indicators
##### OR
>from indictors import `nameOfFunction`

**Step 3 :** pass the data frame and arguments to the function. Here is the full list of indicators currently implemented.
| name of indicator | Functions | Arguement | Return |
|-------------------|-----------|-----------|--------|
| Average True Range (ATR) | atr() | data , period | Series |
| Double Exponential Moving Average (DEMA) | dema() | data , period, column, adjust | Series |
| Exponential Weighted Moving Average (EWMA) or Exponential Moving Average(EMA) | ema() | data , period, column, adjust| Series |
| Kaufman Efficiency indicator (KER) or (ER) | er() | data , period , column | Series |
| Elastic Volume Weighted Moving Average (EVWMA) | evwma() | data , period | Series |
| Elastic Volume Weighted Moving average convergence divergence (EV_MACD) | ev_macd() | data , period_fast, period_slow, signal, adjust | Series |
| Fractal Adaptive Moving Average (FRAMA) | frma() | data , period , batch | Series |
| Hull Moving Average (HMA) | hma() | data , period | Series |
| Kaufman's Adaptive Moving Average (KAMA) | kama() | data , er_ , ema_fast , ema_slow , period , column | Series |
| Moving average convergence divergence (MACD) | macd() | data , period_fast , period_slow, signal, column, adjust | Series |
| Market momentum (MOM) | mom() | data , period, column | Series |
| Pivot Camarilla |  pivot_camarilla() | data | Series |
| Pivot Classic | pivot_classic() | data | Series |
| Pivot Demark | pivot_demark() | data | Series |
| Pivot Fibonacci | pivot_fibonacci | data | Series |
| Pivot Traditional | pivot_traditional() | data | Series |
| Pivot Woodie | pivot_woodie() | data | Series |
| Percentage Price Oscillator (PPO) | ppo() | data , period_fast, period_slow, signal, column, adjust | Series |
| Relative Strength Index (RSI) | rsi() | data , period, column | Series |
| Rate of Change (ROC) | roc() | data , period,column | Series |
| Simple moving average (SMA) or moving average (MA) | sma() | data , period,column | Series |
| Simple moving median (SMM) or moving median (MM) | smm() | data , period, column | Series |
| Simple smoothed moving average (SSMA) or smoothed moving average() | ssma() | data , period, column, adjust | Series |
| Triple Exponential Moving Average (TEMA) | tema() | data , period, column, adjust | Series |
| Typical Price (TP) | tp() | data | Series |
| True Range (TR) | tr() | data | Series |
| Triangular Moving Average (TRIMA) or (TMA) | tma() | data , period, adjust | Series |
| Triple Exponential Average (TRIX) | trix() | data , period, adjust, column| Series |
| Volume Adjusted Moving Average (VAMA) | vama() | data , period, column | Series |
| Volume Weighted Average Price (VWAP) | vwap() | data | Series |
| Volume Weighted Moving average convergence divergence(VWMACD) | vw_macd() | data , period_fast, period_slow, signal, column, adjust | Series |
| Volume Weighted Moving Average (VWMA) | vwma() | data , period, column, adjust | Series |
| Weighted Moving Average (WMA) | wma() | data , period, column | Series |
| Zero Lag Exponential Moving Average (ZLEMA) | zlema() | data , period, adjust, column | Series |

## Dependencies
* Python
* Pandas
* Numpy
