import pandas as pd
import os
import plotly.graph_objs as go
from typing import List
from module_pattern.utils_chanlun.objects import BI, FX, RawBar, NewBar, BiHub
from module_pattern.utils_chanlun.enum import Mark, Direction
from module_pattern.utils_chanlun.utils_plot import kline_pro, KlineChart
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

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

def convert_bi_list_to_df(bi_list):
    """Convert a list of BI objects to a DataFrame"""
    # Define the data types for each column
    data_types = {
        # 'Date': 'object',
        'Idx': 'int64',
        'pv_type': 'object',
        'factor_value': 'float64',
    }

    # Create an empty DataFrame with specified data types
    df_PV_bi = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in data_types.items()})

    # Manually add the first factor/Fx
    df_PV_bi.at[bi_list[0].sdt, 'Idx'] = -1
    if bi_list[0].direction == Direction.Up:
        df_PV_bi.at[bi_list[0].sdt, 'pv_type'] = 'Valley'
        df_PV_bi.at[bi_list[0].sdt, 'factor_value'] = bi_list[0].low
    elif bi_list[0].direction == Direction.Down:
        df_PV_bi.at[bi_list[0].sdt, 'pv_type'] = 'Peak'
        df_PV_bi.at[bi_list[0].sdt, 'factor_value'] = bi_list[0].high

    # Populate the rest DataFrame using bi ends
    for bi in bi_list:
        df_PV_bi.at[bi.edt, 'Idx'] = -1
        if bi.direction == Direction.Up:
            df_PV_bi.at[bi.edt, 'pv_type'] = 'Peak'
            df_PV_bi.at[bi.edt, 'factor_value'] = bi.high

        elif bi.direction == Direction.Down:
            df_PV_bi.at[bi.edt, 'pv_type'] = 'Valley'
            df_PV_bi.at[bi.edt, 'factor_value'] = bi.low

    return df_PV_bi

def find_hubs(df_PV_segments):

    # Create a list to store the hubs
    list_hubs = []

    # Define a function to check if three bis overlap
    def check_new_hub_formation(factor_0_type, factor_0, factor_1, factor_2, factor_3):
        hub_high = None
        hub_low = None
        is_new_hub = False
        if factor_0_type == 'Valley':   ## for a down hub
            is_new_hub = max(factor_0, factor_2) < min(factor_1, factor_3)
            if is_new_hub:
                hub_high = min(factor_1, factor_3)
                hub_low = max(factor_0, factor_2)
        elif factor_0_type == 'Peak':  ## for a up hub
            is_new_hub = min(factor_0, factor_2) > max(factor_1, factor_3)
            if is_new_hub:
                hub_high = min(factor_0, factor_2)
                hub_low = max(factor_1, factor_3)
        return is_new_hub, hub_high, hub_low

    def check_current_hub_belonging(cur_hub, factor_cur_type, factor_cur_value):
        if factor_cur_type == 'Valley':
            if factor_cur_value > cur_hub['high']:
                return False
            else:
                return True
        elif factor_cur_type == 'Peak':
            if factor_cur_value < cur_hub['low']:
                return False
            else:
                return True

    # initialize state variables
    last_included_factor_idx = -1
    in_hub = False

    for i in range(0, len(df_PV_segments) - 3):

        print('Processing factor: ' + str(i))
        if i < last_included_factor_idx:
            # Skip this iteration if it's part of an already processed hub
            continue

        # Get the current factor and the next three factors
        factor_0 = df_PV_segments.iloc[i]['factor_value']
        factor_1 = df_PV_segments.iloc[i + 1]['factor_value']
        factor_2 = df_PV_segments.iloc[i + 2]['factor_value']
        factor_3 = df_PV_segments.iloc[i + 3]['factor_value']
        factor_0_idx = df_PV_segments.index[i]
        factor_1_idx = df_PV_segments.index[i + 1]
        factor_2_idx = df_PV_segments.index[i + 2]
        factor_3_idx = df_PV_segments.index[i + 3]
        factor_0_type = df_PV_segments.iloc[i]['pv_type']

        # Case 1 - if not in a hub, then check if a new hub is formed
        if not in_hub:

            # Determine if the first three lines (current and next two factors) form a hub
            is_new_hub, hub_high, hub_low = check_new_hub_formation(factor_0_type, factor_0, factor_1, factor_2, factor_3)

            # if no new hub can be formed, continue to the next iteration
            if not is_new_hub:
                continue  # Skip to next iteration if not a hub

            # otherwise, define the new hub
            else:

                # if a new hub can be formed, first check its direction
                if len(list_hubs) == 0:
                    cur_hub_direction = 'down' if factor_0_type =='Valley' else 'up'
                else:
                    if hub_low > list_hubs[-1]['low']:
                        cur_hub_direction = 'up'
                    elif hub_high < list_hubs[-1]['high']:
                        cur_hub_direction = 'down'

                # verify the start factor fits the hub direction
                # if wrong, then continue to the next candle
                if (cur_hub_direction == 'up') & (factor_0_type == 'Valley') or\
                    (cur_hub_direction == 'down') & (factor_0_type == 'Peak'):
                    continue

                else:
                    # If a new hub is formed, then set the state variables
                    in_hub = True
                    cur_hub = {'start_idx': factor_0_idx,
                               'end_idx': factor_3_idx,
                               'all_idx': [factor_0_idx, factor_1_idx, factor_2_idx, factor_3_idx],
                               'num_segments': 3,
                               'direction': cur_hub_direction,
                               'high': hub_high,
                               'low': hub_low}

                # continue to check if the next few factors belong to this hub
                for j in range(i + 4, len(df_PV_segments)):
                    factor_cur_value = df_PV_segments.iloc[j]['factor_value']
                    factor_cur_idx = df_PV_segments.index[j]
                    factor_cur_type = df_PV_segments.iloc[j]['pv_type']

                    # if the current factor belongs to the current hub, then update the hub
                    if check_current_hub_belonging(cur_hub, factor_cur_type, factor_cur_value):
                        cur_hub['end_idx'] = factor_cur_idx
                        cur_hub['num_segments'] += 1
                        cur_hub['all_idx'].append(factor_cur_idx)

                    else:
                        # hub breaks
                        in_hub = False

                        # check where the hub should break based on the hub direction
                        if cur_hub_direction == 'down':
                            if factor_cur_type == 'Valley':
                                # down hub should end at peak, so drop the last factor
                                cur_hub['end_idx'] = df_PV_segments.index[j - 3]
                                cur_hub['num_segments'] -= 2
                                last_included_factor_idx = j - 1
                            elif factor_cur_type == 'Peak':
                                cur_hub['end_idx'] = df_PV_segments.index[j - 2]
                                cur_hub['num_segments'] -= 1
                                last_included_factor_idx = j - 1
                        elif cur_hub_direction == 'up':
                            if factor_cur_type == 'Peak':
                                # up hub should end at valley, so drop the last factor
                                cur_hub['end_idx'] = df_PV_segments.index[j - 3]
                                cur_hub['num_segments'] -= 2
                                last_included_factor_idx = j - 1
                            elif factor_cur_type == 'Valley':
                                cur_hub['end_idx'] = df_PV_segments.index[j - 2]
                                cur_hub['num_segments'] -= 1
                                last_included_factor_idx = j - 1

                        # if less than 3 lines in hub
                        if cur_hub['num_segments'] <= 2:
                            # hub not valid
                            break
                        else:
                            # append the hub and break
                            list_hubs.append(cur_hub)
                            break

                    # case if it is the last factor
                    if j == len(df_PV_segments) - 1:
                        in_hub = False
                        list_hubs.append(cur_hub)
                        last_included_factor_idx = j
                        # debug_logging(f'Hub breaks at {factor_cur_idx}')
                        break

    print(list_hubs)
    df_hubs = pd.DataFrame(list_hubs)

    # --- Visualization
    # if debug_plot:
        # debug_plot_hubs_using_HL(df_PV_segments, df_HL, df_hubs)
        # debug_plot_hubs_using_OHLC(df_PV_segments, df_OHLC, df_hubs)
    # fig_hubs = debug_plot_hubs(df_PV_segments, df_HL, df_OHLC, df_hubs)

    return df_hubs

def pattern_setup_trending_hubs_pull_back(df_hubs, cur_price):
    """ This function identifies the pullback trading setup for trending hubs"""

    # Extract the last two hubs
    if len(df_hubs) < 2:
        return 0, 'No valid setup'
    else:
        hub_cur = df_hubs.iloc[-1]
        hub_prev = df_hubs.iloc[-2]


    # --- Check the relation between the two hubs
    # Case 1 - the last hub is higher than the previous hub (up trend)
    msg = 'No valid setup'
    if hub_cur['low'] > hub_prev['high']:  # up trending hubs
        if cur_price < hub_cur['high']:   # loose condition, if price lower than the current high, OK
            msg = 'Pullback long setup'
            return 1, msg  # pullback long setup
        else:
            return 0, msg  # no valid setup
    # Case 2 - the last hub is lower than the previous one (down trend)
    elif hub_cur['high'] < hub_prev['low']:  # down trending hubs
        if cur_price > hub_cur['low']:  # loose condition, if price higher than the current low, OK
            msg = 'Pullback short setup'
            return -1, msg  # pullback short setup
        else:
            return 0, msg  # no valid setup

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

class CZSC:
    def __init__(self,
                 bars: List[RawBar],
                 symbol: 'str',
                 freq: 'str',
                 max_bi_num=100,

                 ):
        """

        :param bars: K线数据
        :param max_bi_num: 最大允许保留的笔数量
        :param get_signals: 自定义的信号计算函数
        """
        # self.verbose = True
        self.max_bi_num = max_bi_num
        self.bars_raw: List[RawBar] = []  # 原始K线序列
        self.bars_ubi: List[NewBar] = []  # 未完成笔的无包含K线序列
        self.bi_list: List[BI] = []
        self.bi_hubs: List[BiHub] = []
        self.symbol = symbol
        self.freq = freq
        # self.signals = None
        # self.chart = None

        # self.fx_list: List[FX] = []
        # self.line_list: List[Line] = []
        # self.line_hubs: List[LineHub] = []

        # # cache 是信号计算过程的缓存容器，需要信号计算函数自行维护
        # self.cache = OrderedDict()

        # 完成笔的处理
        for bar in bars:
            self.update(bar)

        # convert bi_list to dataframe
        bi_list = self.bi_list.copy()

        # convert bi to dataframe
        df_bi = convert_bi_list_to_df(bi_list)

        # manually add the ending bi
        if df_bi.iloc[-1]['pv_type'] == 'Peak':
            df_bi.loc[bars[-1].dt, 'Idx'] = -1
            df_bi.loc[bars[-1].dt, 'pv_type'] = 'Valley'
            df_bi.loc[bars[-1].dt, 'factor_value'] = bars[-1].low
        elif df_bi.iloc[-1]['pv_type'] == 'Valley':
            df_bi.loc[bars[-1].dt, 'Idx'] = -1
            df_bi.loc[bars[-1].dt, 'pv_type'] = 'Peak'
            df_bi.loc[bars[-1].dt, 'factor_value'] = bars[-1].high

        # calculate bi hubs 笔中枢的处理
        df_hubs = find_hubs(df_bi)
        self.df_bi_hubs = df_hubs

        # identify trading pattern
        cur_price = bars[-1].close
        decision, msg = pattern_setup_trending_hubs_pull_back(df_hubs, cur_price)
        self.decision = decision
        self.msg = msg

        # # plot using to_echarts
        self.chart = self.to_echarts()
        self.chart = self.to_plotly()
        self.chart.show()


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
        for idx in range(len(self.df_bi_hubs)):
            bi_hub = self.df_bi_hubs.iloc[idx]


            try:
                rect = go.layout.Shape(
                    type="rect",
                    x0=bi_hub.start_idx, x1=bi_hub.end_idx,
                    y0=bi_hub.low, y1=bi_hub.high,
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
         name_symbol,
         time_frame,
         num_candles=300,
         # use_high_low=False,
         ):

    # convert DataFrame to bars object
    df_OHLC_mid = df_OHLC_mid[-num_candles:]
    bars = convert_df_to_bars(df_OHLC_mid, time_frame, name_symbol)

    # initilize the ChanAnalysis object
    chanlun = CZSC(bars,
                   symbol=name_symbol,
                   freq=time_frame,
                   )

    # plot using to_echarts
    chanlun.chart = chanlun.to_echarts()
    chanlun.chart = chanlun.to_plotly()
    # chanlun.chart.show()

    return chanlun.decision, chanlun.chart
