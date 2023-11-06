# Kitkat's quant trading system

**1. SUMMARY**

**2. TRADING STRATEGY**

The trading strategy relies on three equally important module, corresponding to three different timeframes.

**2.1. Trading direction module**

At a higher timeframe (12h), trading direction (Long, Short, or Unknown) will be decided, based on:
* Technical analysis of trend
* Market condition indicators
  - Overall market breadth
  - Funding rate
  - Open interest
  - Conditions from other relevant markets 

Input to this module is:
* Most recent N=100 candlesticks (OHLC + volume)
* Other relevant indicators

Output of this module is:
* 1 : Long
* -1 : Short
* 0 : Unknown

**2.2. Pattern recognition module**

At a medium timeframe (1h), the specific trading pattern will be explored. This is what typical trading strategy refers to. Specific strategies could be:
* Chanlun 缠论
* Channel rangebound 通道区间
* Support & Resistance 支持阻力
* Wakeng 挖坑战法

Input to this module is:
* Most recent N=100 candlesticks (OHLC + volume)

Output of this module is:
* 1 : pattern detected (add symbol to close watch for trading entry
* 0 : no pattern 

**2.3. Order entry module**

At a lower timeframe (5min), trading entry will be evaluated. A trade will be executed if certain rules are met. Some example rules could be:
* Divergence signal (RSI, MACD)
* Price action (strong pinbar with high volume)
* New high/low triggers
* Other factors
* ML-based decision tree
