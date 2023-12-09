import numpy as np
import pandas as pd
import os
import traceback
import plotly.graph_objs as go
from typing import List, Callable
from collections import OrderedDict
from datetime import datetime
from module_pattern.utils_chanlun.objects import BI, FakeBI, FX, RawBar, NewBar, Line, BiHub, LineHub, Point
from module_pattern.utils_chanlun.enum import Mark, Direction, Operate
from module_pattern.utils_chanlun.utils_plot import kline_pro, KlineChart


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 200)

def convert_df_to_bars(df_OHLC, time_scale, name_symbol):

    # Create a list to store the bars
    bars_raw = []

    # remove index
    df_OHLC.reset_index(inplace=True)

    # convert dataframe to raw bar elements
    for i, row in df_OHLC.iterrows():
        bar = RawBar(symbol=name_symbol,
                     id=i,
                     dt=row['Date'],
                     freq=time_scale,
                     open=row['Open'],
                     close=row['Close'],
                     high=row['High'],
                     low=row['Low'],
                     vol=row['Volume'],
                     amount=0,
                     )
        bars_raw.append(bar)
    # print(f'Converted DataFrame to {len(bars_raw)} bars')

    return bars_raw


def remove_include(k1: NewBar, k2: NewBar, k3: RawBar):
    """去除包含关系：输入三根k线，其中k1和k2为没有包含关系的K线，k3为原始K线

    处理逻辑如下：

    1. 首先，通过比较k1和k2的高点(high)的大小关系来确定direction的值。如果k1的高点小于k2的高点，
       则设定direction为Up；如果k1的高点大于k2的高点，则设定direction为Down；如果k1和k2的高点相等，
       则创建一个新的K线k4，与k3具有相同的属性，并返回False和k4。

    2. 接下来，判断k2和k3之间是否存在包含关系。如果存在，则根据direction的值进行处理。
        - 如果direction为Up，则选择k2和k3中的较大高点作为新K线k4的高点，较大低点作为低点，较大高点所在的时间戳(dt)作为k4的时间戳。
        - 如果direction为Down，则选择k2和k3中的较小高点作为新K线k4的高点，较小低点作为低点，较小低点所在的时间戳(dt)作为k4的时间戳。
        - 如果direction的值不是Up也不是Down，则抛出ValueError异常。

    3. 根据上述处理得到的高点、低点、开盘价(open_)、收盘价(close)，计算新K线k4的成交量(vol)和成交金额(amount)，
       并将k2中除了与k3时间戳相同的元素之外的其他元素与k3一起作为k4的元素列表(elements)。

    4. 返回一个布尔值和新的K线k4。如果k2和k3之间存在包含关系，则返回True和k4；否则返回False和k4，其中k4与k3具有相同的属性。
    """
    if k1.high < k2.high:
        direction = Direction.Up
    elif k1.high > k2.high:
        direction = Direction.Down
    else:
        k4 = NewBar(symbol=k3.symbol, id=k3.id, freq=k3.freq, dt=k3.dt, open=k3.open,
                    close=k3.close, high=k3.high, low=k3.low, vol=k3.vol, amount=k3.amount, elements=[k3])
        return False, k4

    # 判断 k2 和 k3 之间是否存在包含关系，有则处理
    if (k2.high <= k3.high and k2.low >= k3.low) or (k2.high >= k3.high and k2.low <= k3.low):
        if direction == Direction.Up:
            high = max(k2.high, k3.high)
            low = max(k2.low, k3.low)
            dt = k2.dt if k2.high > k3.high else k3.dt
        elif direction == Direction.Down:
            high = min(k2.high, k3.high)
            low = min(k2.low, k3.low)
            dt = k2.dt if k2.low < k3.low else k3.dt
        else:
            raise ValueError

        open_, close = (high, low) if k3.open > k3.close else (low, high)
        vol = k2.vol + k3.vol
        amount = k2.amount + k3.amount
        # 这里有一个隐藏Bug，len(k2.elements) 在一些及其特殊的场景下会有超大的数量，具体问题还没找到；
        # 临时解决方案是直接限定len(k2.elements)<=100
        elements = [x for x in k2.elements[:100] if x.dt != k3.dt] + [k3]
        k4 = NewBar(symbol=k3.symbol, id=k2.id, freq=k2.freq, dt=dt, open=open_,
                    close=close, high=high, low=low, vol=vol, amount=amount, elements=elements)
        return True, k4
    else:
        k4 = NewBar(symbol=k3.symbol, id=k3.id, freq=k3.freq, dt=k3.dt, open=k3.open,
                    close=k3.close, high=k3.high, low=k3.low, vol=k3.vol, amount=k3.amount, elements=[k3])
        return False, k4


def check_fx(k1: NewBar, k2: NewBar, k3: NewBar):
    """查找分型

    函数计算逻辑：

    1. 如果第二个`NewBar`对象的最高价和最低价都高于第一个和第三个`NewBar`对象的对应价格，那么它被认为是顶分型（G）。
       在这种情况下，函数会创建一个新的`FX`对象，其标记为`Mark.G`，并将其赋值给`fx`。

    2. 如果第二个`NewBar`对象的最高价和最低价都低于第一个和第三个`NewBar`对象的对应价格，那么它被认为是底分型（D）。
       在这种情况下，函数会创建一个新的`FX`对象，其标记为`Mark.D`，并将其赋值给`fx`。

    3. 函数最后返回`fx`，如果没有找到分型，`fx`将为`None`。

    :param k1: 第一个`NewBar`对象
    :param k2: 第二个`NewBar`对象
    :param k3: 第三个`NewBar`对象
    :return: `FX`对象或`None`
    """
    fx = None
    if k1.high < k2.high > k3.high and k1.low < k2.low > k3.low:
        fx = FX(symbol=k1.symbol, dt=k2.dt, mark=Mark.G, high=k2.high, freq=k2.freq,
                low=k2.low, fx=k2.high, elements=[k1, k2, k3])

    if k1.low > k2.low < k3.low and k1.high > k2.high < k3.high:
        fx = FX(symbol=k1.symbol, dt=k2.dt, mark=Mark.D, high=k2.high, freq=k2.freq,
                low=k2.low, fx=k2.low, elements=[k1, k2, k3])

    return fx


def check_fxs(bars: List[NewBar]) -> List[FX]:
    """输入一串无包含关系K线，查找其中所有分型

    函数的主要步骤：

    1. 创建一个空列表`fxs`用于存储找到的分型。
    2. 遍历`bars`列表中的每个元素（除了第一个和最后一个），并对每三个连续的`NewBar`对象调用`check_fx`函数。
    3. 如果`check_fx`函数返回一个`FX`对象，检查它的标记是否与`fxs`列表中最后一个`FX`对象的标记相同。如果相同，记录一个错误日志。
       如果不同，将这个`FX`对象添加到`fxs`列表中。
    4. 最后返回`fxs`列表，它包含了`bars`列表中所有找到的分型。

    这个函数的主要目的是找出`bars`列表中所有的顶分型和底分型，并确保它们是交替出现的。如果发现连续的两个分型标记相同，它会记录一个错误日志。

    :param bars: 无包含关系K线列表
    :return: 分型列表
    """
    fxs = []
    for i in range(1, len(bars) - 1):
        fx = check_fx(bars[i - 1], bars[i], bars[i + 1])
        if isinstance(fx, FX):
            # 默认情况下，fxs本身是顶底交替的，但是对于一些特殊情况下不是这样; 临时强制要求fxs序列顶底交替
            if len(fxs) >= 2 and fx.mark == fxs[-1].mark:
                print(f"check_fxs错误: {bars[i].dt}，{fx.mark}，{fxs[-1].mark}")
            else:
                fxs.append(fx)
    return fxs


def check_bi(bars: List[NewBar], benchmark=None):
    """输入一串无包含关系K线，查找其中的一笔

    :param bars: 无包含关系K线列表
    :param benchmark: 当下笔能量的比较基准
    :return:
    """
    # min_bi_len = envs.get_min_bi_len()
    min_bi_len = 6
    fxs = check_fxs(bars)
    if len(fxs) < 2:
        return None, bars

    fx_a = fxs[0]
    if fx_a.mark == Mark.D:
        direction = Direction.Up
        fxs_b = (x for x in fxs if x.mark == Mark.G and x.dt > fx_a.dt and x.fx > fx_a.fx)
        fx_b = max(fxs_b, key=lambda fx: fx.high, default=None)

    elif fx_a.mark == Mark.G:
        direction = Direction.Down
        fxs_b = (x for x in fxs if x.mark == Mark.D and x.dt > fx_a.dt and x.fx < fx_a.fx)
        fx_b = min(fxs_b, key=lambda fx: fx.low, default=None)

    else:
        raise ValueError

    if fx_b is None:
        return None, bars

    bars_a = [x for x in bars if fx_a.elements[0].dt <= x.dt <= fx_b.elements[2].dt]
    bars_b = [x for x in bars if x.dt >= fx_b.elements[0].dt]

    # 判断fx_a和fx_b价格区间是否存在包含关系
    ab_include = (fx_a.high > fx_b.high and fx_a.low < fx_b.low) or (fx_a.high < fx_b.high and fx_a.low > fx_b.low)

    # 判断当前笔的涨跌幅是否超过benchmark的一定比例
    # if benchmark and abs(fx_a.fx - fx_b.fx) > benchmark * envs.get_bi_change_th():
    if benchmark and abs(fx_a.fx - fx_b.fx) > benchmark * -1.0:
        power_enough = True
    else:
        power_enough = False

    # 成笔的条件：1）顶底分型之间没有包含关系；2）笔长度大于等于min_bi_len 或 当前笔的涨跌幅已经够大
    if (not ab_include) and (len(bars_a) >= min_bi_len or power_enough):
        fxs_ = [x for x in fxs if fx_a.elements[0].dt <= x.dt <= fx_b.elements[2].dt]
        bi = BI(symbol=fx_a.symbol, freq='', id=0, fx_a=fx_a, fx_b=fx_b, fxs=fxs_, direction=direction, bars=bars_a)
        return bi, bars_b
    else:
        return None, bars


def generate_line_by_bi(line_list: List[Line], bi_list: List[BI]):
    """根据笔列表生成线段，每次生成一根线段

    :param line_list: 线段列表
    :param bi_list: 笔列表
    :return: line_list, bi_list: 更新后的线段列表和笔列表
    """
    # 如果没有初始线段
    if len(line_list) == 0:
        for i in range(1, len(bi_list) - 1):
            bi1 = bi_list[i - 1]
            bi2 = bi_list[i]
            bi3 = bi_list[i + 1]
            if bi2.power < bi1.power + bi3.power:
                high = max(bi1.high, bi3.high)
                low = min(bi1.low, bi3.low)
                line_list.append(
                    Line(symbol=bi1.symbol, freq=bi1.freq, id=-1, direction=bi1.direction, start_dt=bi1.fx_a.dt,
                         end_dt=bi3.fx_b.dt, high=high, low=low, power=round(abs(high - low), 2)))
                bi_list = [x for x in bi_list if x.fx_a.dt > bi3.fx_a.dt]
                break

    if len(line_list) == 0:
        return line_list, bi_list

    # 进行线段的延伸
    last_line = line_list[-1]
    for bi in bi_list[1::2]:
        if (bi.direction == 'up' and bi.high > last_line.high) or (bi.direction == 'down' and bi.low < last_line.low):
            # 延伸last_line
            last_line.end_dt = bi.fx_b.dt
            if bi.direction == 'up':
                last_line.high = bi.high
            if bi.direction == 'down':
                last_line.low = bi.low
    bi_list = [x for x in bi_list if x.fx_a.dt >= last_line.end_dt]

    new_line, bi_list = check_line_by_bi(bi_list)
    if new_line is not None:
        line_list.append(new_line)

    return line_list, bi_list


def check_line_by_bi(bi_list: List[BI]):
    """输入一串笔序列，查找其中的一根线段"""
    if not bi_list or len(bi_list) < 3:
        return None, bi_list

    bi0 = bi_list[0]
    new_line = None
    for cur in bi_list[::2]:
        if (cur.direction == 'up' and cur.high > bi0.high) or (cur.direction == 'down' and cur.low < bi0.low):
            high = max(bi0.high, cur.high)
            low = min(bi0.low, cur.low)
            new_line = Line(symbol=cur.symbol, freq=cur.freq, id=-1, direction=bi0.direction, start_dt=bi0.fx_a.dt,
                            end_dt=cur.fx_b.dt, high=high, low=low, power=round(abs(high - low), 2))
            bi_list = [x for x in bi_list if x.fx_a.dt >= new_line.end_dt]
            break

    return new_line, bi_list


def generate_biHub(hubs: List[BiHub], bi_list: List[BI], point_list: List[Point]):
    """根据笔列表生成中枢，每次生成一个中枢

    :param hubs: 中枢列表
    :param bi_list: 笔列表
    :return: hubs, bi_list: 更新后的中枢列表和笔列表
    """
    if len(bi_list) < 3:
        return hubs, bi_list, point_list

    # 获取上一个中枢或者第一个中枢
    if not hubs or len(hubs) < 1:
        last_hub, bi_list = check_biHub(bi_list, None)
        if last_hub:
            hubs.append(last_hub)
    else:
        last_hub = hubs[-1]

    if not last_hub:
        return hubs, bi_list, point_list

    # 上一个中枢延伸
    pos = 0
    while pos < len(bi_list):
        if not last_hub.leave:
            last_hub.leave = bi_list[pos]
        if last_hub.leave.fx_a.dt == bi_list[pos].fx_a.dt:
            pos += 1
            continue
        if last_hub.ZD > bi_list[pos].high or last_hub.ZG < bi_list[pos].low:
            if last_hub.ZD > bi_list[pos].high:
                # 中枢结束，形成三卖
                fx = bi_list[pos].fx_b if bi_list[pos].direction == 'up' else bi_list[pos].fx_a
                if len(point_list) == 0 or (len(point_list) > 0 and point_list[-1].dt < fx.dt):
                    point_list.append(
                        Point(id=-1, symbol=fx.symbol, freq=fx.freq, dt=fx.dt, type='S3',
                              high=fx.elements[1].high, low=fx.elements[1].low))
            elif last_hub.ZG < bi_list[pos].low:
                # 中枢结束，形成三买
                fx = bi_list[pos].fx_b if bi_list[pos].direction == 'down' else bi_list[pos].fx_a
                if len(point_list) == 0 or (len(point_list) > 0 and point_list[-1].dt < fx.dt):
                    point_list.append(
                        Point(id=-1, symbol=fx.symbol, freq=fx.freq, dt=fx.dt, type='B3',
                              high=fx.elements[1].high, low=fx.elements[1].low))
            break
        last_hub.elements.append(bi_list[pos])
        last_hub.GG = max([x.high for x in last_hub.elements])
        last_hub.DD = min([x.low for x in last_hub.elements])
        last_hub.leave = bi_list[pos + 1] if pos < len(bi_list) - 1 else None
        pos += 2

    # 计算当前中枢
    cur_hub, bi_list = check_biHub([x for x in bi_list if x.fx_a.dt >= last_hub.elements[-1].fx_b.dt], last_hub)
    if cur_hub and not cur_hub.entry:
        cur_hub.entry = last_hub.leave

    if cur_hub:
        hubs.append(cur_hub)

    return hubs, bi_list, point_list


def check_biHub(bi_list: List[BI], last_hub):
    """输入一串笔，查找其中的第一个中枢

    :param last_hub:
    :param bi_list: 笔列表
    :return: hub, bi_list: 查找到的中枢，和更新后的笔列表
    """
    start_idx = -1
    for i in range(len(bi_list) - 3):
        bi1 = bi_list[i]
        bi3 = bi_list[i + 2]
        zg = min(bi1.high, bi3.high)
        zd = max(bi1.low, bi3.low)
        if zg > zd:
            # 检验中枢方向是否正确
            if last_hub is not None:
                if (bi1.direction == 'down' and zg < last_hub.ZG) or (bi1.direction == 'up' and zd > last_hub.ZD):
                    continue
            # 记录中枢开始位置
            start_idx = i
            break
    if start_idx < 0:
        return None, bi_list

    bi1 = bi_list[start_idx]
    bi3 = bi_list[start_idx + 2]
    entry = None if start_idx == 0 else bi_list[start_idx - 1]
    if entry is None and last_hub is not None:
        entry = last_hub.elements[-1]
    leave = None if start_idx >= len(bi_list) - 3 else bi_list[start_idx + 3]

    hub = BiHub(id=-1, symbol=bi1.symbol, freq=bi1.freq, ZG=min(bi1.high, bi3.high), ZD=max(bi1.low, bi3.low),
              GG=max(bi1.high, bi3.high), DD=min(bi1.low, bi3.low), entry=entry, leave=leave, elements=[bi1, bi3])

    bi_list = [x for x in bi_list if x.fx_a.dt >= hub.elements[-1].fx_b.dt]

    return hub, bi_list

def generate_lineHub(hubs: List[LineHub], line_list: List[Line], point_list: List[Point]):
    """根据线段列表生成中枢，每次生成一个中枢

    :param hubs: 中枢列表
    :param line_list: 笔列表
    :return: hubs, line_list: 更新后的中枢列表和笔列表
    """
    if len(line_list) < 3:
        return hubs, line_list, point_list

    # 获取上一个中枢或者第一个中枢
    if not hubs or len(hubs) < 1:
        last_hub, line_list = check_lineHub(line_list, None)
        if last_hub:
            hubs.append(last_hub)
    else:
        last_hub = hubs[-1]

    if not last_hub:
        return hubs, line_list, point_list

    # 上一个中枢延伸
    pos = 0
    while pos < len(line_list):
        if not last_hub.leave:
            last_hub.leave = line_list[pos]
        if last_hub.leave.start_dt == line_list[pos].start_dt:
            pos += 1
            continue
        if last_hub.ZD > line_list[pos].high or last_hub.ZG < line_list[pos].low:
            if last_hub.ZD > line_list[pos].high:
                # 中枢结束，形成三卖
                dt = line_list[pos].end_dt if line_list[pos].direction == 'up' else line_list[pos].start_dt
                if len(point_list) == 0 or (len(point_list) > 0 and point_list[-1].dt < dt):
                    point_list.append(
                        Point(id=-1, symbol=last_hub.symbol, freq=last_hub.freq, dt=dt, type='S3',
                              high=line_list[pos].high, low=line_list[pos].high))
            elif last_hub.ZG < line_list[pos].low:
                # 中枢结束，形成三买
                dt = line_list[pos].end_dt if line_list[pos].direction == 'down' else line_list[pos].start_dt
                if len(point_list) == 0 or (len(point_list) > 0 and point_list[-1].dt < dt):
                    point_list.append(
                        Point(id=-1, symbol=last_hub.symbol, freq=last_hub.freq, dt=dt, type='B3',
                              high=line_list[pos].low, low=line_list[pos].low))
            break
        last_hub.elements.append(line_list[pos])
        last_hub.GG = max([x.high for x in last_hub.elements])
        last_hub.DD = min([x.low for x in last_hub.elements])
        last_hub.leave = line_list[pos + 1] if pos < len(line_list) - 1 else None
        pos += 2

    # 计算当前中枢
    cur_hub, line_list = check_lineHub([x for x in line_list if x.start_dt >= last_hub.elements[-1].end_dt], last_hub)
    if cur_hub and not cur_hub.entry:
        cur_hub.entry = last_hub.leave

    if cur_hub:
        hubs.append(cur_hub)

    return hubs, line_list, point_list


def check_lineHub(line_list: List[Line], last_hub):
    """输入一串线段，查找其中的第一个中枢

    :param last_hub:
    :param line_list: 线段列表
    :return: hub, line_list: 查找到的中枢，和更新后的线段列表
    """
    start_idx = -1
    for i in range(len(line_list) - 3):
        bi1 = line_list[i]
        bi3 = line_list[i + 2]
        zg = min(bi1.high, bi3.high)
        zd = max(bi1.low, bi3.low)
        if zg > zd:
            # 检验中枢方向是否正确
            if last_hub is not None:
                if (bi1.direction == 'down' and zg < last_hub.ZG) or (bi1.direction == 'up' and zd > last_hub.ZD):
                    continue
            # 记录中枢开始位置
            start_idx = i
            break
    if start_idx < 0:
        return None, line_list

    bi1 = line_list[start_idx]
    bi3 = line_list[start_idx + 2]
    entry = None if start_idx == 0 else line_list[start_idx - 1]
    if entry is None and last_hub is not None:
        entry = last_hub.elements[-1]
    leave = None if start_idx >= len(line_list) - 3 else line_list[start_idx + 3]

    hub = LineHub(id=-1, symbol=bi1.symbol, freq=bi1.freq, ZG=min(bi1.high, bi3.high), ZD=max(bi1.low, bi3.low),
              GG=max(bi1.high, bi3.high), DD=min(bi1.low, bi3.low), entry=entry, leave=leave, elements=[bi1, bi3])

    line_list = [x for x in line_list if x.start_dt >= hub.elements[-1].end_dt]

    return hub, line_list


class CZSC:
    def __init__(self,
                 bars: List[RawBar],
                 get_signals=None,
                 max_bi_num=100,
                 ):
        """

        :param bars: K线数据
        :param max_bi_num: 最大允许保留的笔数量
        :param get_signals: 自定义的信号计算函数
        """
        self.verbose = True
        self.max_bi_num = max_bi_num
        self.bars_raw: List[RawBar] = []  # 原始K线序列
        self.bars_ubi: List[NewBar] = []  # 未完成笔的无包含K线序列
        self.bi_list: List[BI] = []
        # self.fx_list: List[FX] = []
        self.line_list: List[Line] = []
        self.bi_hubs: List[BiHub] = []
        self.line_hubs: List[LineHub] = []
        self.symbol = bars[0].symbol
        self.freq = bars[0].freq
        self.signals = None
        self.chart = None
        # cache 是信号计算过程的缓存容器，需要信号计算函数自行维护
        self.cache = OrderedDict()

        # 完成笔的处理
        for bar in bars:
            self.update(bar)

        # 生成笔中枢
        self.bi_hubs, self.bi_list, self.bi_points = generate_biHub(self.bi_hubs, self.bi_list, [])
        print(f'number of bi hubs: {len(self.bi_hubs)}')

        for bi_hub in self.bi_hubs:
            try:
                print(bi_hub.ZG, bi_hub.ZD, bi_hub.entry.edt, bi_hub.leave.sdt)
            except:
                print('error')
            # # print(bi_hub.ZG, bi_hub.ZD)
            # for bi in bi_hub.elements:
            #     print(bi.fx_a.dt, bi.fx_b.dt)

        # # 生成线段
        # self.line_list, self.bi_list = generate_line_by_bi(self.line_list, self.bi_list)
        # for line in self.line_list:
        #     print(line.start_dt, line.end_dt)

        # # 生成线段中枢
        # self.line_hubs, self.line_list, self.line_points = generate_lineHub(self.line_hubs, self.line_list, [])

        # plot using to_echarts
        self.chart = self.to_echarts()

        # plot using plotly
        self.chart = self.to_plotly()
        self.chart.show()

        print('Done!')

    def __repr__(self):
        return "<CZSC~{}~{}>".format(self.symbol, self.freq.value)

    def __update_bi(self):
        bars_ubi = self.bars_ubi
        if len(bars_ubi) < 3:
            return

        # 查找笔
        if not self.bi_list:
            # 第一笔的查找
            fxs = check_fxs(bars_ubi)
            if not fxs:
                return

            fx_a = fxs[0]
            fxs_a = [x for x in fxs if x.mark == fx_a.mark]
            for fx in fxs_a:
                if (fx_a.mark == Mark.D and fx.low <= fx_a.low) \
                        or (fx_a.mark == Mark.G and fx.high >= fx_a.high):
                    fx_a = fx
            bars_ubi = [x for x in bars_ubi if x.dt >= fx_a.elements[0].dt]

            bi, bars_ubi_ = check_bi(bars_ubi)
            if isinstance(bi, BI):
                self.bi_list.append(bi)
            self.bars_ubi = bars_ubi_
            return

        if self.verbose and len(bars_ubi) > 100:
            print(f"{self.symbol} - {self.freq} - {bars_ubi[-1].dt} 未完成笔延伸数量: {len(bars_ubi)}")

        # if envs.get_bi_change_th() > 0.5 and len(self.bi_list) >= 5:
        #     price_seq = [x.power_price for x in self.bi_list[-5:]]
        #     benchmark = min(self.bi_list[-1].power_price, sum(price_seq) / len(price_seq))
        # else:
        #     benchmark = None
        benchmark = None

        bi, bars_ubi_ = check_bi(bars_ubi, benchmark)
        self.bars_ubi = bars_ubi_
        if isinstance(bi, BI):
            self.bi_list.append(bi)

        # 后处理：如果当前笔被破坏，将当前笔的bars与bars_ubi进行合并，并丢弃
        last_bi = self.bi_list[-1]
        bars_ubi = self.bars_ubi
        if (last_bi.direction == Direction.Up and bars_ubi[-1].high > last_bi.high) \
                or (last_bi.direction == Direction.Down and bars_ubi[-1].low < last_bi.low):
            # 当前笔被破坏，将当前笔的bars与bars_ubi进行合并，并丢弃，这里容易出错，多一根K线就可能导致错误
            # 必须是 -2，因为最后一根无包含K线有可能是未完成的
            self.bars_ubi = last_bi.bars[:-2] + [x for x in bars_ubi if x.dt >= last_bi.bars[-2].dt]
            self.bi_list.pop(-1)

    def update(self, bar: RawBar):
        """更新分析结果

        :param bar: 单根K线对象
        """
        # 更新K线序列
        if not self.bars_raw or bar.dt != self.bars_raw[-1].dt:
            self.bars_raw.append(bar)
            last_bars = [bar]
        else:
            # 当前 bar 是上一根 bar 的时间延伸
            self.bars_raw[-1] = bar
            last_bars = self.bars_ubi.pop(-1).raw_bars
            assert bar.dt == last_bars[-1].dt
            last_bars[-1] = bar

        # 去除包含关系
        bars_ubi = self.bars_ubi
        for bar in last_bars:
            if len(bars_ubi) < 2:
                bars_ubi.append(NewBar(symbol=bar.symbol, id=bar.id, freq=bar.freq, dt=bar.dt,
                                       open=bar.open, close=bar.close, amount=bar.amount,
                                       high=bar.high, low=bar.low, vol=bar.vol, elements=[bar]))
            else:
                k1, k2 = bars_ubi[-2:]
                has_include, k3 = remove_include(k1, k2, bar)
                if has_include:
                    bars_ubi[-1] = k3
                else:
                    bars_ubi.append(k3)
        self.bars_ubi = bars_ubi

        # 更新笔
        self.__update_bi()

        # 根据最大笔数量限制完成 bi_list, bars_raw 序列的数量控制
        self.bi_list = self.bi_list[-self.max_bi_num:]
        if self.bi_list:
            sdt = self.bi_list[0].fx_a.elements[0].dt
            s_index = 0
            for i, bar in enumerate(self.bars_raw):
                if bar.dt >= sdt:
                    s_index = i
                    break
            self.bars_raw = self.bars_raw[s_index:]

        # 如果有信号计算函数，则进行信号计算
        # self.signals = self.get_signals(c=self) if self.get_signals else OrderedDict()

    def to_echarts(self, width: str = "1400px", height: str = '580px', bs=[]):
        """绘制K线分析图

        :param width: 宽
        :param height: 高
        :param bs: 交易标记，默认为空
        :return:
        """
        kline = [x.__dict__ for x in self.bars_raw]
        if len(self.bi_list) > 0:
            bi = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx} for x in self.bi_list] + \
                 [{'dt': self.bi_list[-1].fx_b.dt, "bi": self.bi_list[-1].fx_b.fx}]
            fx = [{'dt': x.dt, "fx": x.fx} for x in self.fx_list]
        else:
            bi = []
            fx = []
        chart = kline_pro(kline, bi=bi, fx=fx, width=width, height=height, bs=bs,
                          title="{}-{}".format(self.symbol, ''))
        return chart

    def to_plotly(self):
        """使用 plotly 绘制K线分析图"""
        import pandas as pd

        bi_list = self.bi_list
        df = pd.DataFrame(self.bars_raw)
        kline = KlineChart(n_rows=3, title="{}-{}".format(self.symbol, self.freq))
        kline.add_kline(df, name="")
        # kline.add_sma(df, ma_seq=(5, 10, 21), row=1, visible=True, line_width=1.2)
        # kline.add_sma(df, ma_seq=(34, 55, 89, 144), row=1, visible=False, line_width=1.2)
        kline.add_vol(df, row=2)
        kline.add_macd(df, row=3)

        if len(bi_list) > 0:
            bi1 = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx, "text": x.fx_a.mark.value} for x in bi_list]
            bi2 = [{'dt': bi_list[-1].fx_b.dt, "bi": bi_list[-1].fx_b.fx, "text": bi_list[-1].fx_b.mark.value[0]}]
            bi = pd.DataFrame(bi1 + bi2)
            fx = pd.DataFrame([{'dt': x.dt, "fx": x.fx} for x in self.fx_list])
            # kline.add_scatter_indicator(fx['dt'], fx['fx'], name="分型", row=1, line_width=1)
            kline.add_scatter_indicator(bi['dt'], bi['bi'], name="笔", text=bi['text'], row=1, line_width=2)

        # Check if there are already shapes in the figure and keep them
        existing_shapes = list(kline.fig.layout.shapes) if kline.fig.layout.shapes is not None else []

        # Loop through each bi_hub and add rectangles
        for bi_hub in self.bi_hubs:
            try:
                rect = go.layout.Shape(
                    type="rect",
                    x0=bi_hub.entry.sdt, x1=bi_hub.leave.sdt,
                    y0=bi_hub.ZG, y1=bi_hub.ZD,
                    line=dict(width=2),
                    fillcolor="LightSkyBlue",
                    opacity=0.5,
                    layer="below"  # Ensures the shape is below the data points
                )
                existing_shapes.append(rect)
            except AttributeError:
                # Skip if any of the required attributes are missing
                continue

        # Update the figure with the new shapes
        kline.fig.update_layout(shapes=existing_shapes)


        return kline.fig

    def open_in_browser(self, width: str = "1400px", height: str = '580px'):
        """直接在浏览器中打开分析结果

        :param width: 图表宽度
        :param height: 图表高度
        :return:
        """
        home_path = os.path.expanduser("~")
        file_html = os.path.join(home_path, "temp_czsc.html")
        chart = self.to_echarts(width, height)
        chart.render(file_html)
        # webbrowser.open(file_html)

    @property
    def last_bi_extend(self):
        """判断最后一笔是否在延伸中，True 表示延伸中"""
        if self.bi_list[-1].direction == Direction.Up \
                and max([x.high for x in self.bars_ubi]) > self.bi_list[-1].high:
            return True

        if self.bi_list[-1].direction == Direction.Down \
                and min([x.low for x in self.bars_ubi]) < self.bi_list[-1].low:
            return True

        return False

    @property
    def finished_bis(self) -> List[BI]:
        """已完成的笔"""
        if not self.bi_list:
            return []
        if len(self.bars_ubi) < 5:
            return self.bi_list[:-1]
        return self.bi_list

    @property
    def ubi_fxs(self) -> List[FX]:
        """bars_ubi 中的分型"""
        if not self.bars_ubi:
            return []
        else:
            return check_fxs(self.bars_ubi)

    @property
    def ubi(self):
        """Unfinished Bi，未完成的笔"""
        ubi_fxs = self.ubi_fxs
        if not self.bars_ubi or not self.bi_list or not ubi_fxs:
            return None

        bars_raw = [y for x in self.bars_ubi for y in x.raw_bars]
        # 获取最高点和最低点，以及对应的时间
        high_bar = max(bars_raw, key=lambda x: x.high)
        low_bar = min(bars_raw, key=lambda x: x.low)
        direction = Direction.Up if self.bi_list[-1].direction == Direction.Down else Direction.Down

        bi = {
            "symbol": self.symbol,
            "direction": direction,
            "high": high_bar.high,
            "low": low_bar.low,
            "high_bar": high_bar,
            "low_bar": low_bar,
            "bars": self.bars_ubi,
            "raw_bars": bars_raw,
            "fxs": ubi_fxs,
            "fx_a": ubi_fxs[0],
        }
        return bi

    @property
    def fx_list(self) -> List[FX]:
        """分型列表，包括 bars_ubi 中的分型"""
        fxs = []
        for bi_ in self.bi_list:
            fxs.extend(bi_.fxs[1:])
        ubi = self.ubi_fxs
        for x in ubi:
            if not fxs or x.dt > fxs[-1].dt:
                fxs.append(x)
        return fxs




def main(df_OHLC_mid,
         name_symbol='BTCUSDT',
         time_frame='12h',
         num_candles=400,
         debug_plot=False,
         use_high_low=False):

    # convert DataFrame to bars
    df_OHLC_mid = df_OHLC_mid[-num_candles:]
    bars = convert_df_to_bars(df_OHLC_mid, time_frame, name_symbol)

    # initilize the ChanAnalysis object
    chan_analysis = CZSC(bars)


