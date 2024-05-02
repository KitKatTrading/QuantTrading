# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2021/3/10 12:21
describe: 常用对象结构
"""
import math
import hashlib
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
# from loguru import logger
# from deprecated import deprecated
from typing import List, Callable, Dict
# from czsc.enum import Mark, Direction, Freq, Operate
# from czsc.utils.corr import single_linear


# coding: utf-8
from enum import Enum


class Operate(Enum):
    # 持有状态
    HL = "持多"  # Hold Long
    HS = "持空"  # Hold Short
    HO = "持币"  # Hold Other

    # 多头操作
    LO = "开多"  # Long Open
    LE = "平多"  # Long Exit

    # 空头操作
    SO = "开空"  # Short Open
    SE = "平空"  # Short Exit

    def __str__(self):
        return self.value


class Mark(Enum):
    D = "底分型"
    G = "顶分型"

    def __str__(self):
        return self.value


class Direction(Enum):
    Up = "向上"
    Down = "向下"

    def __str__(self):
        return self.value


class Freq(Enum):
    Tick = "Tick"
    F1 = "1分钟"
    F2 = "2分钟"
    F3 = "3分钟"
    F4 = "4分钟"
    F5 = "5分钟"
    F6 = "6分钟"
    F10 = "10分钟"
    F12 = "12分钟"
    F15 = "15分钟"
    F20 = "20分钟"
    F30 = "30分钟"
    F60 = "60分钟"
    F120 = "120分钟"
    D = "日线"
    W = "周线"
    M = "月线"
    S = "季线"
    Y = "年线"

    def __str__(self):
        return self.value


# @deprecated(version="1.0.0", reason="请使用 RawBar")
@dataclass
class Tick:
    symbol: str
    name: str = ""
    price: float = 0
    vol: float = 0


@dataclass
class RawBar:
    """原始K线元素"""

    symbol: str
    id: int  # id 必须是升序
    dt: datetime
    freq: Freq
    open: float
    close: float
    high: float
    low: float
    vol: float
    amount: float
    cache: dict = field(default_factory=dict)  # cache 用户缓存，一个最常见的场景是缓存技术指标计算结果

    @property
    def upper(self):
        """上影"""
        return self.high - max(self.open, self.close)

    @property
    def lower(self):
        """下影"""
        return min(self.open, self.close) - self.low

    @property
    def solid(self):
        """实体"""
        return abs(self.open - self.close)


@dataclass
class NewBar:
    """去除包含关系后的K线元素"""

    symbol: str
    id: int  # id 必须是升序
    dt: datetime
    freq: Freq
    open: float
    close: float
    high: float
    low: float
    vol: float
    amount: float
    elements: List = field(default_factory=list)  # 存入具有包含关系的原始K线
    cache: dict = field(default_factory=dict)  # cache 用户缓存

    @property
    def raw_bars(self):
        return self.elements


@dataclass
class FX:
    symbol: str
    dt: datetime
    mark: Mark
    high: float
    low: float
    fx: float
    freq: None
    elements: List = field(default_factory=list)
    cache: dict = field(default_factory=dict)  # cache 用户缓存

    @property
    def new_bars(self):
        """构成分型的无包含关系K线"""
        return self.elements

    @property
    def raw_bars(self):
        """构成分型的原始K线"""
        res = []
        for e in self.elements:
            res.extend(e.raw_bars)
        return res

    @property
    def power_str(self):
        assert len(self.elements) == 3
        k1, k2, k3 = self.elements

        if self.mark == Mark.D:
            if k3.close > k1.high:
                x = "强"
            elif k3.close > k2.high:
                x = "中"
            else:
                x = "弱"
        else:
            assert self.mark == Mark.G
            if k3.close < k1.low:
                x = "强"
            elif k3.close < k2.low:
                x = "中"
            else:
                x = "弱"
        return x

    @property
    def power_volume(self):
        """成交量力度"""
        assert len(self.elements) == 3
        return sum([x.vol for x in self.elements])

    @property
    def has_zs(self):
        """构成分型的三根无包含K线是否有重叠中枢"""
        assert len(self.elements) == 3
        zd = max([x.low for x in self.elements])
        zg = min([x.high for x in self.elements])
        return zg >= zd


@dataclass
class FakeBI:
    """虚拟笔：主要为笔的内部分析提供便利"""

    symbol: str
    sdt: datetime
    edt: datetime
    direction: Direction
    high: float
    low: float
    power: float
    cache: dict = field(default_factory=dict)  # cache 用户缓存


def create_fake_bis(fxs: List[FX]) -> List[FakeBI]:
    """创建 fake_bis 列表

    :param fxs: 分型序列，必须顶底分型交替
    :return: fake_bis
    """
    if len(fxs) % 2 != 0:
        fxs = fxs[:-1]

    fake_bis = []
    for i in range(1, len(fxs)):
        fx1 = fxs[i - 1]
        fx2 = fxs[i]
        assert fx1.mark != fx2.mark
        if fx1.mark == Mark.D:
            fake_bi = FakeBI(
                symbol=fx1.symbol,
                sdt=fx1.dt,
                edt=fx2.dt,
                direction=Direction.Up,
                high=fx2.high,
                low=fx1.low,
                power=round(fx2.high - fx1.low, 2),
            )
        elif fx1.mark == Mark.G:
            fake_bi = FakeBI(
                symbol=fx1.symbol,
                sdt=fx1.dt,
                edt=fx2.dt,
                direction=Direction.Down,
                high=fx1.high,
                low=fx2.low,
                power=round(fx1.high - fx2.low, 2),
            )
        else:
            raise ValueError
        fake_bis.append(fake_bi)
    return fake_bis


@dataclass
class BI:
    symbol: str
    freq: str
    id: 0
    fx_a: FX    # 笔开始的分型
    fx_b: FX    # 笔结束的分型
    fxs: List   # 笔内部的分型列表
    direction: Direction
    bars: List[NewBar] = field(default_factory=list)
    cache: dict = field(default_factory=dict)  # cache 用户缓存

    def __post_init__(self):
        self.sdt = self.fx_a.dt
        self.edt = self.fx_b.dt

    def __repr__(self):
        return (
            f"BI(symbol={self.symbol}, sdt={self.sdt}, edt={self.edt}, "
            f"direction={self.direction}, high={self.high}, low={self.low})"
        )

    def get_cache_with_default(self, key, default: Callable):
        """带有默认值计算的缓存读取

        :param key: 缓存 key
        :param default: 如果没有缓存数据，用来计算默认值并更新缓存的函数
        :return:
        """
        cache = self.cache if self.cache else {}
        value = cache.get(key, None)
        if not value:
            value = default()
            cache[key] = value
            self.cache = cache
        return value

    # 定义一些附加属性，用的时候才会计算，提高效率
    # ======================================================================
    @property
    def fake_bis(self):
        """笔的内部分型连接得到近似次级别笔列表"""

        def __default():
            return create_fake_bis(self.fxs)

        return self.get_cache_with_default("fake_bis", __default)

    @property
    def high(self):
        def __default():
            return max(self.fx_a.high, self.fx_b.high)

        return self.get_cache_with_default("high", __default)

    @property
    def low(self):
        return min(self.fx_a.low, self.fx_b.low)

    @property
    def power(self):
        return self.power_price

    @property
    def power_price(self):
        """价差力度"""
        return round(abs(self.fx_b.fx - self.fx_a.fx), 2)

    @property
    def power_volume(self):
        """成交量力度"""
        return sum([x.vol for x in self.bars[1:-1]])

    @property
    def change(self):
        """笔的涨跌幅"""
        c = round((self.fx_b.fx - self.fx_a.fx) / self.fx_a.fx, 4)
        return c

    @property
    def length(self):
        """笔的无包含关系K线数量"""
        return len(self.bars)

    @property
    def rsq(self):
        """笔的原始K线 close 单变量线性回归 r2"""
        value = self.get_price_linear("close")
        return round(value["r2"], 4)

    @property
    def raw_bars(self):
        """构成笔的原始K线序列"""

        def __default():
            value = []
            for bar in self.bars[1:-1]:
                value.extend(bar.raw_bars)
            return value

        return self.get_cache_with_default("raw_bars", __default)

    @property
    def hypotenuse(self):
        """笔的斜边长度"""
        return pow(pow(self.power_price, 2) + pow(len(self.raw_bars), 2), 1 / 2)

    @property
    def angle(self):
        """笔的斜边与竖直方向的夹角，角度越大，力度越大"""
        return round(math.asin(self.power_price / self.hypotenuse) * 180 / 3.14, 2)


@dataclass
class Line:
    """线段"""
    symbol: str
    freq: str
    id: int
    direction: str
    start_dt: datetime
    end_dt: datetime
    high: float = None
    low: float = None
    power: float = None
    # seqs: List[Seq] = None
    # fx_a: SeqFX = None  # 线段开始的分型
    # fx_b: SeqFX = None  # 线段结束的分型


@dataclass
class BiHub:
    """笔构成的中枢"""
    id: int
    symbol: str
    freq: str
    ZG: float
    ZD: float
    GG: float
    DD: float
    entry: BI = None
    leave: BI = None
    elements: List[BI] = None # 奇数位的笔


@dataclass
class LineHub:
    """线段构成的中枢"""
    id: int
    symbol: str
    freq: str
    ZG: float
    ZD: float
    GG: float
    DD: float
    entry: Line = None
    leave: Line = None
    elements: List[Line] = None # 奇数位的笔


@dataclass
class Point:
    """买卖点"""
    id: int
    symbol: str
    freq: str
    dt: datetime
    type: str
    high: float
    low: float

#
# @dataclass
# class ZS:
#     """中枢对象，主要用于辅助信号函数计算"""
#
#     bis: List[BI]
#     cache: dict = field(default_factory=dict)  # cache 用户缓存
#
#     def __post_init__(self):
#         self.symbol = self.bis[0].symbol
#
#     @property
#     def sdt(self):
#         """中枢开始时间"""
#         return self.bis[0].sdt
#
#     @property
#     def edt(self):
#         """中枢结束时间"""
#         return self.bis[-1].edt
#
#     @property
#     def sdir(self):
#         """中枢第一笔方向，sdir 是 start direction 的缩写"""
#         return self.bis[0].direction
#
#     @property
#     def edir(self):
#         """中枢倒一笔方向，edir 是 end direction 的缩写"""
#         return self.bis[-1].direction
#
#     @property
#     def zz(self):
#         """中枢中轴"""
#         return self.zd + (self.zg - self.zd) / 2
#
#     @property
#     def gg(self):
#         """中枢最高点"""
#         return max([x.high for x in self.bis])
#
#     @property
#     def zg(self):
#         """中枢上沿"""
#         return min([x.high for x in self.bis[:3]])
#
#     @property
#     def dd(self):
#         """中枢最低点"""
#         return min([x.low for x in self.bis])
#
#     @property
#     def zd(self):
#         """中枢下沿"""
#         return max([x.low for x in self.bis[:3]])
#
#     @property
#     def is_valid(self):
#         """中枢是否有效"""
#         if self.zg < self.zd:
#             return False
#
#         for bi in self.bis:
#             # 中枢内的笔必须与中枢的上下沿有交集
#             if (
#                 self.zg >= bi.high >= self.zd
#                 or self.zg >= bi.low >= self.zd
#                 or bi.high >= self.zg > self.zd >= bi.low
#             ):
#                 continue
#             else:
#                 return False
#
#         return True
#
#     def __repr__(self):
#         return (
#             f"ZS(sdt={self.sdt}, sdir={self.sdir}, edt={self.edt}, edir={self.edir}, "
#             f"len_bis={len(self.bis)}, zg={self.zg}, zd={self.zd}, "
#             f"gg={self.gg}, dd={self.dd}, zz={self.zz})"
#         )
#
#
# @dataclass
# class Signal:
#     signal: str = ""
#
#     # score 取值在 0~100 之间，得分越高，信号越强
#     score: int = 0
#
#     # k1, k2, k3 是信号名称
#     k1: str = "任意"  # k1 一般是指明信号计算的K线周期，如 60分钟，日线，周线等
#     k2: str = "任意"  # k2 一般是记录信号计算的参数
#     k3: str = "任意"  # k3 用于区分信号，必须具有唯一性，推荐使用信号分类和开发日期进行标记
#
#     # v1, v2, v3 是信号取值
#     v1: str = "任意"
#     v2: str = "任意"
#     v3: str = "任意"
#
#     # 任意 出现在模板信号中可以指代任何值
#
#     def __post_init__(self):
#         if not self.signal:
#             self.signal = f"{self.k1}_{self.k2}_{self.k3}_{self.v1}_{self.v2}_{self.v3}_{self.score}"
#         else:
#             (
#                 self.k1,
#                 self.k2,
#                 self.k3,
#                 self.v1,
#                 self.v2,
#                 self.v3,
#                 score,
#             ) = self.signal.split("_")
#             self.score = int(score)
#
#         if self.score > 100 or self.score < 0:
#             raise ValueError("score 必须在0~100之间")
#
#     def __repr__(self):
#         return f"Signal('{self.signal}')"
#
#     @property
#     def key(self) -> str:
#         """获取信号名称"""
#         key = ""
#         for k in [self.k1, self.k2, self.k3]:
#             if k != "任意":
#                 key += k + "_"
#         return key.strip("_")
#
#     @property
#     def value(self) -> str:
#         """获取信号值"""
#         return f"{self.v1}_{self.v2}_{self.v3}_{self.score}"
#
#     def is_match(self, s: dict) -> bool:
#         """判断信号是否与信号列表中的值匹配
#
#         代码的执行逻辑如下：
#
#         接收一个字典 s 作为参数，该字典包含了所有信号的信息。从字典 s 中获取名称为 key 的信号的值 v。
#         如果 v 不存在，则抛出异常。从信号的值 v 中解析出 v1、v2、v3 和 score 四个变量。
#
#         如果当前信号的得分 score 大于等于目标信号的得分 self.score，则继续执行，否则返回 False。
#         如果当前信号的第一个值 v1 等于目标信号的第一个值 self.v1 或者目标信号的第一个值为 "任意"，则继续执行，否则返回 False。
#         如果当前信号的第二个值 v2 等于目标信号的第二个值 self.v2 或者目标信号的第二个值为 "任意"，则继续执行，否则返回 False。
#         如果当前信号的第三个值 v3 等于目标信号的第三个值 self.v3 或者目标信号的第三个值为 "任意"，则返回 True，否则返回 False。
#
#         :param s: 所有信号字典
#         :return: bool
#         """
#         key = self.key
#         v = s.get(key, None)
#         if not v:
#             raise ValueError(f"{key} 不在信号列表中")
#
#         v1, v2, v3, score = v.split("_")
#         if int(score) >= self.score:
#             if v1 == self.v1 or self.v1 == "任意":
#                 if v2 == self.v2 or self.v2 == "任意":
#                     if v3 == self.v3 or self.v3 == "任意":
#                         return True
#         return False
#
#
# @dataclass
# class Factor:
#     # signals_all 必须全部满足的信号，至少需要设定一个信号
#     signals_all: List[Signal]
#
#     # signals_any 满足其中任一信号，允许为空
#     signals_any: List[Signal] = field(default_factory=list)
#
#     # signals_not 不能满足其中任一信号，允许为空
#     signals_not: List[Signal] = field(default_factory=list)
#
#     name: str = ""
#
#     def __post_init__(self):
#         if not self.signals_all:
#             raise ValueError("signals_all 不能为空")
#         _fatcor = self.dump()
#         _fatcor.pop("name")
#         sha256 = hashlib.sha256(str(_fatcor).encode("utf-8")).hexdigest().upper()[:8]
#         self.name = f"{self.name}#{sha256}" if self.name else sha256
#
#     @property
#     def unique_signals(self) -> List[str]:
#         """获取 Factor 的唯一信号列表"""
#         signals = []
#         signals.extend(self.signals_all)
#         if self.signals_any:
#             signals.extend(self.signals_any)
#         if self.signals_not:
#             signals.extend(self.signals_not)
#         signals = {x.signal if isinstance(x, Signal) else x for x in signals}
#         return list(signals)
#
#     def is_match(self, s: dict) -> bool:
#         """判断 factor 是否满足"""
#         if self.signals_not:
#             for signal in self.signals_not:
#                 if signal.is_match(s):
#                     return False
#
#         for signal in self.signals_all:
#             if not signal.is_match(s):
#                 return False
#
#         if not self.signals_any:
#             return True
#
#         for signal in self.signals_any:
#             if signal.is_match(s):
#                 return True
#         return False
#
#     def dump(self) -> dict:
#         """将 Factor 对象转存为 dict"""
#         signals_all = [x.signal for x in self.signals_all]
#         signals_any = [x.signal for x in self.signals_any] if self.signals_any else []
#         signals_not = [x.signal for x in self.signals_not] if self.signals_not else []
#
#         raw = {
#             "name": self.name,
#             "signals_all": signals_all,
#             "signals_any": signals_any,
#             "signals_not": signals_not,
#         }
#         return raw
#
#     @classmethod
#     def load(cls, raw: dict):
#         """从 dict 中创建 Factor
#
#         :param raw: 样例如下
#             {'name': '单测',
#             'signals_all': ['15分钟_倒0笔_方向_向上_其他_其他_0', '15分钟_倒0笔_长度_大于5_其他_其他_0'],
#             'signals_any': [],
#             'signals_not': []}
#
#         :return:
#         """
#         signals_any = [Signal(x) for x in raw.get("signals_any", [])]
#         signals_not = [Signal(x) for x in raw.get("signals_not", [])]
#
#         fa = Factor(
#             name=raw.get("name", ""),
#             signals_all=[Signal(x) for x in raw["signals_all"]],
#             signals_any=signals_any,
#             signals_not=signals_not,
#         )
#         return fa

#
# @dataclass
# class Event:
#     operate: Operate
#
#     # 多个信号组成一个因子，多个因子组成一个事件。
#     # 单个事件是一系列同类型因子的集合，事件中的任一因子满足，则事件为真。
#     factors: List[Factor]
#
#     # signals_all 必须全部满足的信号，允许为空
#     signals_all: List[Signal] = field(default_factory=list)
#
#     # signals_any 满足其中任一信号，允许为空
#     signals_any: List[Signal] = field(default_factory=list)
#
#     # signals_not 不能满足其中任一信号，允许为空
#     signals_not: List[Signal] = field(default_factory=list)
#
#     name: str = ""
#
#     def __post_init__(self):
#         if not self.factors:
#             raise ValueError("factors 不能为空")
#         _event = self.dump()
#         _event.pop("name")
#         sha256 = hashlib.sha256(str(_event).encode("utf-8")).hexdigest().upper()[:8]
#         if self.name:
#             self.name = f"{self.name}#{sha256}"
#         else:
#             self.name = f"{self.operate.value}#{sha256}"
#         self.sha256 = sha256
#
#     @property
#     def unique_signals(self) -> List[str]:
#         """获取 Event 的唯一信号列表"""
#         signals = []
#         if self.signals_all:
#             signals.extend(self.signals_all)
#         if self.signals_any:
#             signals.extend(self.signals_any)
#         if self.signals_not:
#             signals.extend(self.signals_not)
#
#         for factor in self.factors:
#             signals.extend(factor.unique_signals)
#
#         signals = {x.signal if isinstance(x, Signal) else x for x in signals}
#         return list(signals)
#
#     def get_signals_config(self, signals_module: str = "czsc.signals") -> List[Dict]:
#         """获取事件的信号配置"""
#         from czsc.traders.sig_parse import get_signals_config
#
#         return get_signals_config(self.unique_signals, signals_module)
#
#     def is_match(self, s: dict):
#         """判断 event 是否满足
#
#         代码的执行逻辑如下：
#
#         1. 首先判断 signals_not 中的信号是否得到满足，如果满足任意一个信号，则直接返回 False，表示事件不满足。
#         2. 接着判断 signals_all 中的信号是否全部得到满足，如果有任意一个信号不满足，则直接返回 False，表示事件不满足。
#         3. 然后判断 signals_any 中的信号是否有一个得到满足，如果一个都不满足，则直接返回 False，表示事件不满足。
#         4. 最后判断因子是否满足，顺序遍历因子列表，找到第一个满足的因子就退出，并返回 True 和该因子的名称，表示事件满足。
#         5. 如果遍历完所有因子都没有找到满足的因子，则返回 False，表示事件不满足。
#         """
#         # 首先判断 event 层面的信号是否得到满足
#         if self.signals_not:
#             # 满足任意一个，直接返回 False
#             for signal in self.signals_not:
#                 if signal.is_match(s):
#                     return False, None
#
#         if self.signals_all:
#             # 任意一个不满足，直接返回 False
#             for signal in self.signals_all:
#                 if not signal.is_match(s):
#                     return False, None
#
#         if self.signals_any:
#             one_match = False
#             for signal in self.signals_any:
#                 if signal.is_match(s):
#                     one_match = True
#                     break
#             # 一个都不满足，直接返回 False
#             if not one_match:
#                 return False, None
#
#         # 判断因子是否满足，顺序遍历，找到第一个满足的因子就退出
#         # 因子放入事件中时，建议因子列表按关注度从高到低排序
#         for factor in self.factors:
#             if factor.is_match(s):
#                 return True, factor.name
#
#         return False, None
#
#     def dump(self) -> dict:
#         """将 Event 对象转存为 dict"""
#         signals_all = [x.signal for x in self.signals_all] if self.signals_all else []
#         signals_any = [x.signal for x in self.signals_any] if self.signals_any else []
#         signals_not = [x.signal for x in self.signals_not] if self.signals_not else []
#         factors = [x.dump() for x in self.factors]
#
#         raw = {
#             "name": self.name,
#             "operate": self.operate.value,
#             "signals_all": signals_all,
#             "signals_any": signals_any,
#             "signals_not": signals_not,
#             "factors": factors,
#         }
#         return raw
#
#     @classmethod
#     def load(cls, raw: dict):
#         """从 dict 中创建 Event
#
#         :param raw: 样例如下
#                         {'name': '单测',
#                          'operate': '开多',
#                          'factors': [{'name': '测试',
#                              'signals_all': ['15分钟_倒0笔_长度_大于5_其他_其他_0'],
#                              'signals_any': [],
#                              'signals_not': []}],
#                          'signals_all': ['15分钟_倒0笔_方向_向上_其他_其他_0'],
#                          'signals_any': [],
#                          'signals_not': []}
#         :return:
#         """
#         # 检查输入参数是否合法
#         assert (
#             raw["operate"] in Operate.__dict__["_value2member_map_"]
#         ), f"operate {raw['operate']} not in Operate"
#         assert raw["factors"], "factors can not be empty"
#
#         e = Event(
#             name=raw.get("name", ""),
#             operate=Operate.__dict__["_value2member_map_"][raw["operate"]],
#             factors=[Factor.load(x) for x in raw["factors"]],
#             signals_all=[Signal(x) for x in raw.get("signals_all", [])],
#             signals_any=[Signal(x) for x in raw.get("signals_any", [])],
#             signals_not=[Signal(x) for x in raw.get("signals_not", [])],
#         )
#         return e
#
