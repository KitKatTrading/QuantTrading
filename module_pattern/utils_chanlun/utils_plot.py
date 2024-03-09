from pyecharts import options as opts
from pyecharts.charts import HeatMap, Kline, Line, Bar, Scatter, Grid, Boxplot
from pyecharts.commons.utils import JsCode
from typing import List
import numpy as np
import pandas as pd
import os
import webbrowser
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def SMA(close: np.array, timeperiod=5):
    """简单移动平均

    https://baike.baidu.com/item/%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF/217887

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < timeperiod:
            seq = close[0: i+1]
        else:
            seq = close[i - timeperiod + 1: i + 1]
        res.append(seq.mean())
    return np.array(res, dtype=np.double).round(4)

def EMA(close: np.array, timeperiod=5):
    """
    https://baike.baidu.com/item/EMA/12646151

    :param close: np.array
        收盘价序列
    :param timeperiod: int
        均线参数
    :return: np.array
    """
    res = []
    for i in range(len(close)):
        if i < 1:
            res.append(close[i])
        else:
            ema = (2 * close[i] + res[i-1] * (timeperiod-1)) / (timeperiod+1)
            res.append(ema)
    return np.array(res, dtype=np.double).round(4)

def MACD(close: np.array, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD 异同移动平均线
    https://baike.baidu.com/item/MACD%E6%8C%87%E6%A0%87/6271283

    :param close: np.array
        收盘价序列
    :param fastperiod: int
        快周期，默认值 12
    :param slowperiod: int
        慢周期，默认值 26
    :param signalperiod: int
        信号周期，默认值 9
    :return: (np.array, np.array, np.array)
        diff, dea, macd
    """
    ema12 = EMA(close, timeperiod=fastperiod)
    ema26 = EMA(close, timeperiod=slowperiod)
    diff = ema12 - ema26
    dea = EMA(diff, timeperiod=signalperiod)
    macd = (diff - dea) * 2
    return diff.round(4), dea.round(4), macd.round(4)

def KDJ(close: np.array, high: np.array, low: np.array):
    """

    :param close: 收盘价序列
    :param high: 最高价序列
    :param low: 最低价序列
    :return:
    """
    n = 9
    hv = []
    lv = []
    for i in range(len(close)):
        if i < n:
            h_ = high[0: i+1]
            l_ = low[0: i+1]
        else:
            h_ = high[i - n + 1: i + 1]
            l_ = low[i - n + 1: i + 1]
        hv.append(max(h_))
        lv.append(min(l_))

    hv = np.around(hv, decimals=2)
    lv = np.around(lv, decimals=2)
    rsv = np.where(hv == lv, 0, (close - lv) / (hv - lv) * 100)

    k = []
    d = []
    j = []
    for i in range(len(rsv)):
        if i < n:
            k_ = rsv[i]
            d_ = k_
        else:
            k_ = (2 / 3) * k[i-1] + (1 / 3) * rsv[i]
            d_ = (2 / 3) * d[i-1] + (1 / 3) * k_

        k.append(k_)
        d.append(d_)
        j.append(3 * k_ - 2 * d_)

    k = np.array(k, dtype=np.double)
    d = np.array(d, dtype=np.double)
    j = np.array(j, dtype=np.double)
    return k.round(4), d.round(4), j.round(4)

def RSQ(close: [np.array, list]) -> float:
    """拟合优度 R SQuare

    :param close: 收盘价序列
    :return:
    """
    x = list(range(len(close)))
    y = np.array(close)
    x_squred_sum = sum([x1 * x1 for x1 in x])
    xy_product_sum = sum([x[i] * y[i] for i in range(len(x))])
    num = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    delta = float(num * x_squred_sum - x_sum * x_sum)
    if delta == 0:
        return 0
    y_intercept = (1 / delta) * (x_squred_sum * y_sum - x_sum * xy_product_sum)
    slope = (1 / delta) * (num * xy_product_sum - x_sum * y_sum)

    y_mean = np.mean(y)
    ss_tot = sum([(y1 - y_mean) * (y1 - y_mean) for y1 in y]) + 0.00001
    ss_err = sum([(y[i] - slope * x[i] - y_intercept) * (y[i] - slope * x[i] - y_intercept) for i in range(len(x))])
    rsq = 1 - ss_err / ss_tot

    return round(rsq, 4)

def heat_map(data: List[dict],
             x_label: List[str] = None,
             y_label: List[str] = None,
             title: str = "热力图",
             width: str = "900px",
             height: str = "680px") -> HeatMap:
    """绘制热力图

    :param data: 用于绘制热力图的数据，示例如下
        [{'x': '0hour', 'y': '0day', 'heat': 11},
         {'x': '0hour', 'y': '1day', 'heat': 40},
         {'x': '0hour', 'y': '2day', 'heat': 38},
         {'x': '0hour', 'y': '3day', 'heat': 36},
         {'x': '0hour', 'y': '4day', 'heat': 11}]
    :param x_label: x轴标签
    :param y_label: y轴标签
    :param title: 图表标题
    :param width: 图表宽度
    :param height: 图表高度
    :return: 图表
    """

    value = [[s['x'], s['y'], s['heat']] for s in data]
    heat = [s['heat'] for s in data]

    if not x_label:
        x_label = sorted(list(set([s['x'] for s in data])))

    if not y_label:
        y_label = sorted(list(set([s['y'] for s in data])))

    vis_map_opts = opts.VisualMapOpts(pos_left="90%", pos_top="20%", min_=min(heat), max_=max(heat))
    title_opts = opts.TitleOpts(title=title)
    init_opts = opts.InitOpts(page_title=title, width=width, height=height)
    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0], pos_top="96%", pos_bottom="0%",
                                  range_start=80, range_end=100)
    legend_opts = opts.LegendOpts(is_show=False)

    hm = HeatMap(init_opts=init_opts)
    hm.add_xaxis(x_label)
    hm.add_yaxis("heat", y_label, value, label_opts=opts.LabelOpts(is_show=True, position="inside"))
    hm.set_global_opts(title_opts=title_opts, visualmap_opts=vis_map_opts, legend_opts=legend_opts,
                       xaxis_opts=opts.AxisOpts(grid_index=0), datazoom_opts=[dz_inside, dz_slider])
    return hm


def kline_pro(kline: List[dict],
              fx: List[dict] = None,
              bi: List[dict] = None,
              xd: List[dict] = None,
              bs: List[dict] = None,
              title: str = "缠论K线分析",
              width: str = "1400px",
              height: str = '900px') -> Grid:
    """绘制缠中说禅K线分析结果

    :param kline: K线
    :param fx: 分型识别结果
    :param bi: 笔识别结果
        {'dt': Timestamp('2020-11-26 00:00:00'),
          'fx_mark': 'd',
          'start_dt': Timestamp('2020-11-25 00:00:00'),
          'end_dt': Timestamp('2020-11-27 00:00:00'),
          'fx_high': 144.87,
          'fx_low': 138.0,
          'bi': 138.0}
    :param xd: 线段识别结果
    :param bs: 买卖点
    :param title: 图表标题
    :param width: 图表宽度
    :param height: 图表高度
    :return: 用Grid组合好的图表
    """
    # 配置项设置
    # ------------------------------------------------------------------------------------------------------------------
    bg_color = "#1f212d"  # 背景
    up_color = "#F9293E"
    down_color = "#00aa3b"

    init_opts = opts.InitOpts(bg_color=bg_color, width=width, height=height, animation_opts=opts.AnimationOpts(False))
    title_opts = opts.TitleOpts(title=title, pos_top="1%",
                                title_textstyle_opts=opts.TextStyleOpts(color=up_color, font_size=20),
                                subtitle_textstyle_opts=opts.TextStyleOpts(color=down_color, font_size=12))

    label_not_show_opts = opts.LabelOpts(is_show=False)
    legend_not_show_opts = opts.LegendOpts(is_show=False)
    red_item_style = opts.ItemStyleOpts(color=up_color)
    green_item_style = opts.ItemStyleOpts(color=down_color)
    k_style_opts = opts.ItemStyleOpts(color=up_color, color0=down_color, border_color=up_color,
                                      border_color0=down_color, opacity=0.8)

    legend_opts = opts.LegendOpts(is_show=True, pos_top="1%", pos_left="30%", item_width=14, item_height=8,
                                  textstyle_opts=opts.TextStyleOpts(font_size=12, color="#0e99e2"))
    brush_opts = opts.BrushOpts(tool_box=["rect", "polygon", "keep", "clear"],
                                x_axis_index="all", brush_link="all",
                                out_of_brush={"colorAlpha": 0.1}, brush_type="lineX")

    axis_pointer_opts = opts.AxisPointerOpts(is_show=True, link=[{"xAxisIndex": "all"}])

    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0, 1, 2], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0, 1, 2], pos_top="96%",
                                  pos_bottom="0%", range_start=80, range_end=100)

    yaxis_opts = opts.AxisOpts(is_scale=True,
                               axislabel_opts=opts.LabelOpts(color="#c7c7c7", font_size=8, position="inside"))

    grid0_xaxis_opts = opts.AxisOpts(type_="category", grid_index=0, axislabel_opts=label_not_show_opts,
                                     split_number=20, min_="dataMin", max_="dataMax",
                                     is_scale=True, boundary_gap=False,
                                     axisline_opts=opts.AxisLineOpts(is_on_zero=False))

    tool_tip_opts = opts.TooltipOpts(
        trigger="axis",
        axis_pointer_type="cross",
        background_color="rgba(245, 245, 245, 0.8)",
        border_width=1,
        border_color="#ccc",
        position=JsCode("""
                    function (pos, params, el, elRect, size) {
    					var obj = {top: 10};
    					obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
    					return obj;
    				}
                    """),
        textstyle_opts=opts.TextStyleOpts(color="#000"),
    )

    # 数据预处理
    # ------------------------------------------------------------------------------------------------------------------
    dts = [x['dt'] for x in kline]
    # k_data = [[x['open'], x['close'], x['low'], x['high']] for x in kline]
    k_data = [opts.CandleStickItem(name=i, value=[x['open'], x['close'], x['low'], x['high']])
              for i, x in enumerate(kline)]

    vol = []
    for i, row in enumerate(kline):
        item_style = red_item_style if row['close'] > row['open'] else green_item_style
        bar = opts.BarItem(name=i, value=row['vol'], itemstyle_opts=item_style, label_opts=label_not_show_opts)
        vol.append(bar)

    close = np.array([x['close'] for x in kline], dtype=np.double)
    diff, dea, macd = MACD(close)

    ma5 = SMA(close, timeperiod=5)
    ma34 = SMA(close, timeperiod=34)
    ma233 = SMA(close, timeperiod=233)

    macd_bar = []
    for i, v in enumerate(macd.tolist()):
        item_style = red_item_style if v > 0 else green_item_style
        bar = opts.BarItem(name=i, value=round(v, 4), itemstyle_opts=item_style,
                           label_opts=label_not_show_opts)
        macd_bar.append(bar)

    diff = diff.round(4)
    dea = dea.round(4)

    # K 线主图
    # ------------------------------------------------------------------------------------------------------------------
    chart_k = Kline()
    chart_k.add_xaxis(xaxis_data=dts)
    chart_k.add_yaxis(series_name="Kline", y_axis=k_data, itemstyle_opts=k_style_opts)

    chart_k.set_global_opts(
        legend_opts=legend_opts,
        datazoom_opts=[dz_inside, dz_slider],
        yaxis_opts=yaxis_opts,
        tooltip_opts=tool_tip_opts,
        axispointer_opts=axis_pointer_opts,
        brush_opts=brush_opts,
        title_opts=title_opts,
        xaxis_opts=grid0_xaxis_opts
    )

    # 均线图
    # ------------------------------------------------------------------------------------------------------------------
    chart_ma = Line()
    chart_ma.add_xaxis(xaxis_data=dts)

    ma_keys = {"MA5": ma5, "MA34": ma34, "MA233": ma233}
    ma_colors = ["#39afe6", "#da6ee8", "#00940b"]
    for i, (name, ma) in enumerate(ma_keys.items()):
        chart_ma.add_yaxis(series_name=name, y_axis=ma, is_smooth=True,
                           symbol_size=0, label_opts=label_not_show_opts,
                           linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color=ma_colors[i]))

    chart_ma.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
    chart_k = chart_k.overlap(chart_ma)

    # 缠论结果
    # ------------------------------------------------------------------------------------------------------------------
    if fx:
        fx_dts = [x['dt'] for x in fx]
        fx_val = [x['fx'] for x in fx]
        chart_fx = Scatter()
        chart_fx.add_xaxis(fx_dts)
        chart_fx.add_yaxis(series_name="FX", y_axis=fx_val,
                           symbol="circle", symbol_size=6, label_opts=label_not_show_opts,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(152, 147, 193, 1.0)", ))

        chart_fx.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_fx)

    if bi:
        bi_dts = [x['dt'] for x in bi]
        bi_val = [x['bi'] for x in bi]
        chart_bi = Line()
        chart_bi.add_xaxis(bi_dts)
        chart_bi.add_yaxis(series_name="BI", y_axis=bi_val,
                           symbol="diamond", symbol_size=10, label_opts=label_not_show_opts,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(184, 117, 225, 1.0)", ),
                           linestyle_opts=opts.LineStyleOpts(width=1.5))

        chart_bi.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_bi)

    if xd:
        xd_dts = [x['dt'] for x in xd]
        xd_val = [x['xd'] for x in xd]
        chart_xd = Line()
        chart_xd.add_xaxis(xd_dts)
        chart_xd.add_yaxis(series_name="XD", y_axis=xd_val, symbol="triangle", symbol_size=10,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(37, 141, 54, 1.0)", ))

        chart_xd.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_xd)

    if bs:
        b_dts = [x['dt'] for x in bs if x['mark'] == 'buy']
        if len(b_dts) > 0:
            b_val = [x['price'] for x in bs if x['mark'] == 'buy']
            chart_b = Scatter()
            chart_b.add_xaxis(b_dts)
            chart_b.add_yaxis(series_name="BUY", y_axis=b_val, symbol="arrow", symbol_size=8,
                              itemstyle_opts=opts.ItemStyleOpts(color="#f31e1e", ))

            chart_b.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
            chart_k = chart_k.overlap(chart_b)

        s_dts = [x['dt'] for x in bs if x['mark'] == 'sell']
        if len(s_dts) > 0:
            s_val = [x['price'] for x in bs if x['mark'] == 'sell']
            chart_s = Scatter()
            chart_s.add_xaxis(s_dts)
            chart_s.add_yaxis(series_name="SELL", y_axis=s_val, symbol="pin", symbol_size=12,
                              itemstyle_opts=opts.ItemStyleOpts(color="#45b97d", ))

            chart_s.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
            chart_k = chart_k.overlap(chart_s)

    # 成交量图
    # ------------------------------------------------------------------------------------------------------------------
    chart_vol = Bar()
    chart_vol.add_xaxis(dts)
    chart_vol.add_yaxis(series_name="Volume", y_axis=vol, bar_width='60%')
    chart_vol.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=1,
            axislabel_opts=opts.LabelOpts(is_show=True, font_size=8, color="#9b9da9"),
        ),
        yaxis_opts=yaxis_opts, legend_opts=legend_not_show_opts,
    )

    # MACD图
    # ------------------------------------------------------------------------------------------------------------------
    chart_macd = Bar()
    chart_macd.add_xaxis(dts)
    chart_macd.add_yaxis(series_name="MACD", y_axis=macd_bar, bar_width='60%')
    chart_macd.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=2,
            axislabel_opts=opts.LabelOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            grid_index=2,
            split_number=4,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=True, color="#c7c7c7"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )

    line = Line()
    line.add_xaxis(dts)
    line.add_yaxis(series_name="DIFF", y_axis=diff, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#da6ee8"))
    line.add_yaxis(series_name="DEA", y_axis=dea, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#39afe6"))

    chart_macd = chart_macd.overlap(line)

    grid0_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="12%", height="58%")
    grid1_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="74%", height="8%")
    grid2_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="86%", height="10%")

    grid_chart = Grid(init_opts)
    grid_chart.add(chart_k, grid_opts=grid0_opts)
    grid_chart.add(chart_vol, grid_opts=grid1_opts)
    grid_chart.add(chart_macd, grid_opts=grid2_opts)
    return grid_chart


def box_plot(data: dict,
             title: str = "箱线图",
             width: str = "900px",
             height: str = "680px") -> Boxplot:
    """

    :param data: 数据
        样例：
        data = {
            "expr 0": [960, 850, 830, 880],
            "expr 1": [960, 850, 830, 880],
        }
    :param title:
    :param width:
    :param height:
    :return:
    """
    x_data = []
    y_data = []
    for k, v in data.items():
        x_data.append(k)
        y_data.append(v)

    init_opts = opts.InitOpts(page_title=title, width=width, height=height)

    chart = Boxplot(init_opts=init_opts)
    chart.add_xaxis(xaxis_data=x_data)
    chart.add_yaxis(series_name="", y_axis=y_data)
    chart.set_global_opts(title_opts=opts.TitleOpts(pos_left="center", title=title),
                          tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
                          xaxis_opts=opts.AxisOpts(
                              type_="category",
                              boundary_gap=True,
                              splitarea_opts=opts.SplitAreaOpts(is_show=False),
                              axislabel_opts=opts.LabelOpts(formatter="{value}"),
                              splitline_opts=opts.SplitLineOpts(is_show=False),
                          ),
                          yaxis_opts=opts.AxisOpts(
                              type_="value",
                              name="",
                              splitarea_opts=opts.SplitAreaOpts(
                                  is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                              )
                          ))
    return chart



class KlineChart:
    """K线绘图工具类

    plotly 参数详解: https://www.jianshu.com/p/4f4daf47cc85

    """

    def __init__(self, n_rows=3, **kwargs):
        """K线绘图工具类

        初始化执行逻辑：

        - 接收一个可选参数 n_rows，默认值为 3。这个参数表示图表中的子图数量。
        - 接收一个可变参数列表 **kwargs，可以传递其他配置参数。
        - 如果没有提供 row_heights 参数，则根据 n_rows 设置默认的行高度。
        - 定义了一些颜色变量：color_red 和 color_green。
        - 使用 make_subplots 函数创建一个具有 n_rows 行和 1 列的子图布局，并设置一些共享属性和间距。
        - 使用 fig.update_yaxes 和 fig.update_xaxes 更新 Y 轴和 X 轴的属性，如显示网格、自动调整范围等。
        - 使用 fig.update_layout 更新整个图形的布局，包括标题、边距、图例位置和样式、背景模板等。
        - 将 fig 对象保存在 self.fig 属性中。

        :param n_rows: 子图数量
        :param kwargs:
        """
        self.n_rows = n_rows
        row_heights = kwargs.get("row_heights", None)
        if not row_heights:
            heights_map = {3: [0.6, 0.2, 0.2], 4: [0.55, 0.15, 0.15, 0.15], 5: [0.4, 0.15, 0.15, 0.15, 0.15]}
            assert self.n_rows in heights_map.keys(), "使用内置高度配置，n_rows 只能是 3, 4, 5"
            row_heights = heights_map[self.n_rows]

        self.color_red = 'rgba(249,41,62,0.7)'
        self.color_green = 'rgba(0,170,59,0.7)'
        self.color_yellow = 'rgba(255,174,66,0.7)'
        fig = make_subplots(rows=self.n_rows, cols=1, shared_xaxes=True, row_heights=row_heights,
                            horizontal_spacing=0, vertical_spacing=0)

        fig = fig.update_yaxes(showgrid=True, zeroline=False, automargin=True,
                               fixedrange=kwargs.get('y_fixed_range', True),
                               showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dot')
        fig = fig.update_xaxes(type='category', rangeslider_visible=False, showgrid=False, automargin=True,
                               showticklabels=False, showspikes=True, spikemode='across', spikesnap='cursor',
                               showline=False, spikedash='dot')

        # https://plotly.com/python/reference/layout/
        fig.update_layout(
            title=dict(text=kwargs.get('title', ''), yanchor='middle', xanchor='center'),
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0   # top margin
            ),
            # https://plotly.com/python/reference/layout/#layout-legend
            # legend=dict(orientation='h', yanchor="top", y=1.05, xanchor="center", x=0, bgcolor='rgba(0,0,0,0)'),
            legend=dict(orientation='h', yanchor="top", y=1.05, xanchor="right", x=0.5, bgcolor='rgba(0,0,0,0)'),
            template="plotly_dark",
            hovermode="x unified",
            hoverlabel=dict(bgcolor='rgba(255,255,255,0.1)', font=dict(size=20)),  # 透明，更容易看清后面k线
            dragmode='pan',
            legend_title_font_color="red",
           )

        self.fig = fig

    def add_kline(self, kline: pd.DataFrame, name: str = "K线", **kwargs):
        """绘制K线

        函数执行逻辑：

        1. 检查 kline 数据框是否包含 'text' 列。如果没有，则添加一个空字符串列。
        2. 使用 go.Candlestick 创建一个K线图，并传入以下参数：
            - x: 日期时间数据
            - open, high, low, close: 开盘价、最高价、最低价和收盘价
            - text: 显示在每个 K 线上的文本标签
            - name: 图例名称
            - showlegend: 是否显示图例
            - increasing_line_color 和 decreasing_line_color: 上涨时的颜色和下跌时的颜色
            - increasing_fillcolor 和 decreasing_fillcolor: 上涨时填充颜色和下跌时填充颜色
            - **kwargs: 可以传递其他自定义参数给 Candlestick 函数。

        3. 将创建的烛台图对象添加到 self.fig 中的第一个子图（row=1, col=1）。
        4. 使用 fig.update_traces 更新所有 traces 的 xaxis 属性为 "x1"。
        """
        if 'text' not in kline.columns:
            kline['text'] = ""

        candle = go.Candlestick(x=kline['dt'], open=kline["open"], high=kline["high"], low=kline["low"],
                                close=kline["close"], text=kline["text"], name=name, showlegend=True,
                                increasing_line_color=self.color_green, decreasing_line_color=self.color_red,
                                increasing_fillcolor=self.color_green, decreasing_fillcolor=self.color_red, **kwargs)
        self.fig.add_trace(candle, row=1, col=1)
        self.fig.update_traces(xaxis="x1")

    def add_vol(self, kline: pd.DataFrame, row=2, **kwargs):
        """绘制成交量图

        函数执行逻辑：

        1. 首先，复制输入的 kline 数据框到 df。
        2. 使用 np.where 函数根据收盘价（df['close']）和开盘价（df['open']）之间的关系为 df 创建一个新列 'vol_color'。
           如果收盘价大于开盘价，则使用红色（self.color_red），否则使用绿色（self.color_green）。
        3. 调用 add_bar_indicator 方法绘制成交量图。传递以下参数：
            - x: 日期时间数据
            - y: 成交量数据
            - color: 根据 'vol_color' 列的颜色
            - name: 图例名称
            - row: 指定要添加指标的子图行数，默认值为 2
            - show_legend: 是否显示图例，默认值为 False
        """
        df = kline.copy()
        df['vol_color'] = np.where(df['close'] > df['open'], self.color_red, self.color_green)
        self.add_bar_indicator(df['dt'], df['vol'], color=df['vol_color'], name="成交量", row=row, show_legend=False)

    def add_sma(self, kline: pd.DataFrame, row=1, ma_seq=(5, 10, 20), visible=False, **kwargs):
        """绘制均线图

        函数执行逻辑：

        1. 复制输入的 kline 数据框到 df。
        2. 获取自定义参数 line_width，默认值为 0.6。
        3. 遍历 ma_seq 中的所有均线周期：
            - 对每个周期使用 pandas rolling 方法计算收盘价的移动平均线。
            - 调用 add_scatter_indicator 方法将移动平均线数据绘制为折线图。传递以下参数：
                - x: 日期时间数据
                - y: 移动平均线数据
                - name: 图例名称，格式为 "MA{ma}"，其中 {ma} 是当前的均线周期。
                - row: 指定要添加指标的子图行数，默认值为 1
                - line_width: 线宽，默认值为 0.6
                - visible: 是否可见，默认值为 False
                - show_legend: 是否显示图例，默认值为 True
        """
        df = kline.copy()
        line_width = kwargs.get('line_width', 0.6)
        for ma in ma_seq:
            self.add_scatter_indicator(df['dt'], df['close'].rolling(ma).mean(), name=f"MA{ma}",
                                       row=row, line_width=line_width, visible=visible, show_legend=True)

    def add_rsi(self, kline: pd.DataFrame, row=3, **kwargs):
        """绘制RSI图

        函数执行逻辑：

        1. 首先，复制输入的 kline 数据框到 df。
        2. 获取自定义参数 rsi_periods，默认值为 14。
        3. 使用 talib 库的 RSI 函数计算 RSI 值（rsi）。
        4. 调用 add_scatter_indicator 方法将 rsi 绘制为折线图。传递以下参数：
            - x: 日期时间数据
            - y: rsi 数据
            - name: 图例名称，为 "RSI"
            - row: 指定要添加指标的子图行数，默认值为 3
            - line_width: 线宽，默认值为 0.6
            - show_legend: 是否显示图例，默认值为 False
        5. 调用 add_line_indicator 方法将 rsi 绘制为水平线。传递以下参数：
            - y: rsi 数据
            - name: 图例名称，为 "RSI"
            - row: 指定要添加指标的子图行数，默认值为 3
            - line_width: 线宽，默认值为 0.6
            - show_legend: 是否显示图例，默认值为 False
        """
        import talib
        df = kline.copy()
        rsi_periods = kwargs.get('rsi_periods', 14)
        line_width = kwargs.get('line_width', 0.6)
        rsi = talib.RSI(df['close'], timeperiod=rsi_periods)
        self.add_bar_indicator(df['dt'], rsi, name="RSI", row=row, color=self.color_yellow, show_legend=True)

        # rsi_6EMA = talib.EMA(rsi, timeperiod=6)
        # rsi_12EMA = talib.EMA(rsi, timeperiod=12)
        # rsi_21EMA = talib.EMA(rsi, timeperiod=21)

        # rsi_offset = rsi - 50
        # rsi_6EMA_offset = rsi_6EMA - 50
        # rsi_12EMA_offset = rsi_12EMA - 50
        # rsi_colors = np.where(rsi_offset > 0, self.color_red, self.color_green)
        # self.add_scatter_indicator(df['dt'], rsi_6EMA_offset, name="rsi_6EMA", row=row,
        #                            line_color='white', show_legend=False, line_width=0.6)
        # self.add_scatter_indicator(df['dt'], rsi_12EMA_offset, name="rsi_12EMA", row=row,
        #                            line_color='yellow', show_legend=False, line_width=0.6)
        # self.add_bar_indicator(df['dt'], rsi_offset, name="RSI", row=row, color=rsi_colors, show_legend=True)
        # self.add_bar_indicator(df['dt'], rsi, name="RSI", row=row, color=rsi_colors, show_legend=True)


    def add_macd(self, kline: pd.DataFrame, row=3, **kwargs):
        """绘制MACD图

        函数执行逻辑：

        1. 首先，复制输入的 kline 数据框到 df。
        2. 获取自定义参数 fastperiod、slowperiod 和 signalperiod。这些参数分别对应于计算 MACD 时使用的快周期、慢周期和信号周期，默认值分别为 12、26 和 9。
        3. 使用 talib 库的 MACD 函数计算 MACD 值（diff, dea, macd）。
        4. 创建一个名为 macd_colors 的 numpy 数组，根据 macd 值大于零的情况设置颜色：大于零使用红色（self.color_red），否则使用绿色（self.color_green）。
        5. 调用 add_scatter_indicator 方法将 diff 和 dea 绘制为折线图。传递以下参数：
            - x: 日期时间数据
            - y: diff 或 dea 数据
            - name: 图例名称，分别为 "DIFF" 和 "DEA"
            - row: 指定要添加指标的子图行数，默认值为 3
            - line_color: 线的颜色，分别为 'white' 和 'yellow'
            - show_legend: 是否显示图例，默认值为 False
            - line_width: 线宽，默认值为 0.6
        6. 调用 add_bar_indicator 方法将 macd 绘制为柱状图。传递以下参数：
            - x: 日期时间数据
            - y: macd 数据
            - name: 图例名称，为 "MACD"
            - row: 指定要添加指标的子图行数，默认值为 3
            - color: 根据 macd_colors 设置颜色
            - show_legend: 是否显示图例，默认值为 False
        """
        df = kline.copy()
        fastperiod = kwargs.get('fastperiod', 12)
        slowperiod = kwargs.get('slowperiod', 26)
        signalperiod = kwargs.get('signalperiod', 9)
        line_width = kwargs.get('line_width', 0.6)
        diff, dea, macd = MACD(df["close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        macd_colors = np.where(macd > 0, self.color_red, self.color_green)
        self.add_scatter_indicator(df['dt'], diff, name="DIFF", row=row,
                                   line_color='white', show_legend=False, line_width=line_width)
        self.add_scatter_indicator(df['dt'], dea, name="DEA", row=row,
                                   line_color='yellow', show_legend=False, line_width=line_width)
        self.add_bar_indicator(df['dt'], macd, name="MACD", row=row, color=macd_colors, show_legend=False)

    def add_indicator(self, dt, scatters: list = None, scatter_names: list = None, bar=None, bar_name='', row=4, **kwargs):
        """绘制曲线叠加bar型指标

        1. 获取自定义参数 line_width，默认值为 0.6。
        2. 如果 scatters（列表）不为空，则遍历 scatters 中的所有散点数据：
            - 对于每个散点数据，调用 add_scatter_indicator 方法将其绘制为折线图。传递以下参数：
                - x: 日期时间数据
                - y: 散点数据
                - name: 图例名称，来自 scatter_names 列表
                - row: 指定要添加指标的子图行数，默认值为 4
                - show_legend: 是否显示图例，默认值为 False
                - line_width: 线宽，默认值为 0.6
        3. 如果 bar 不为空，则使用 np.where 函数根据 bar 值大于零的情况设置颜色：大于零使用红色（self.color_red），否则使用绿色（self.color_green）。
        4. 调用 add_bar_indicator 方法将 bar 绘制为柱状图。传递以下参数：
            - x: 日期时间数据
            - y: bar 数据
            - name: 图例名称，为传入的 bar_name 参数
            - row: 指定要添加指标的子图行数，默认值为 4
            - color: 根据上一步计算的颜色设置
            - show_legend: 是否显示图例，默认值为 False
        """
        line_width = kwargs.get('line_width', 0.6)
        for i, scatter in enumerate(scatters):
            self.add_scatter_indicator(dt, scatter, name=scatter_names[i], row=row, show_legend=False, line_width=line_width)

        if bar:
            bar_colors = np.where(np.array(bar, dtype=np.double) > 0, self.color_red, self.color_green)
            self.add_bar_indicator(dt, bar, name=bar_name, row=row, color=bar_colors, show_legend=False)

    def add_marker_indicator(self, x, y, name: str, row: int, text=None, **kwargs):
        """绘制标记类指标

        函数执行逻辑：

        1. 获取自定义参数 line_color、line_width、hover_template、show_legend 和 visible。
            这些参数分别对应于折线颜色、宽度、鼠标悬停时显示的模板、是否显示图例和是否可见。
        2. 使用给定的 x、y 数据创建一个 go.Scatter 对象（散点图），并传入以下参数：
            - x: 指标的x轴数据
            - y: 指标的y轴数据
            - name: 指标名称
            - text: 文本说明
            - line_width: 线宽
            - line_color: 线颜色
            - hovertemplate: 鼠标悬停时显示的模板
            - showlegend: 是否显示图例
            - visible: 是否可见
            - opacity: 透明度
            - mode: 绘制模式，为 'markers' 表示只绘制标记
            - marker: 标记的样式，包括大小、颜色和符号
        3. 调用 self.fig.add_trace 方法将创建的 go.Scatter 对象添加到指定子图中，并更新所有 traces 的 X 轴属性为 "x1"。

        :param x: 指标的x轴
        :param y: 指标的y轴
        :param name: 指标名称
        :param row: 放入第几个子图
        :param text: 文本说明
        :param kwargs:
        :return:
        """
        line_color = kwargs.get('line_color', None)
        line_width = kwargs.get('line_width', None)
        hover_template = kwargs.get('hover_template', '%{y:.3f}-%{text}')
        show_legend = kwargs.get('show_legend', True)
        visible = True if kwargs.get('visible', True) else 'legendonly'
        color = kwargs.get('color', None)
        tag = kwargs.get('tag', None)
        scatter = go.Scatter(x=x, y=y, name=name, text=text, line_width=line_width, line_color=line_color,
                             hovertemplate=hover_template, showlegend=show_legend, visible=visible, opacity=1.0,
                             mode='markers', marker=dict(size=10, color=color, symbol=tag))

        self.fig.add_trace(scatter, row=row, col=1)
        self.fig.update_traces(xaxis="x1")

    def add_scatter_indicator(self, x, y, name: str, row: int, text=None, **kwargs):
        """绘制线性/离散指标

        绘图API文档：https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html

        函数执行逻辑：

        1. 获取自定义参数 mode、hover_template、show_legend、opacity 和 visible。这些参数分别对应于绘图模式、鼠标悬停时显示的模板、是否显示图例、透明度和是否可见。
        2. 使用给定的 x、y 数据创建一个 go.Scatter 对象（散点图），并传入以下参数：
            - x: 指标的x轴数据
            - y: 指标的y轴数据
            - name: 指标名称
            - text: 文本说明
            - mode: 绘制模式，默认为 'text+lines'，表示同时绘制文本和线条
            - hovertemplate: 鼠标悬停时显示的模板
            - showlegend: 是否显示图例
            - visible: 是否可见
            - opacity: 透明度
        3. 调用 self.fig.add_trace 方法将创建的 go.Scatter 对象添加到指定子图中，并更新所有 traces 的 X 轴属性为 "x1"。

        :param x: 指标的x轴
        :param y: 指标的y轴
        :param name: 指标名称
        :param row: 放入第几个子图
        :param text: 文本说明
        :param kwargs:
        :return:
        """
        mode = kwargs.pop('mode', 'text+lines')
        hover_template = kwargs.pop('hover_template', '%{y:.3f}')
        show_legend = kwargs.pop('show_legend', True)
        opacity = kwargs.pop('opacity', 1.0)
        visible = True if kwargs.pop('visible', True) else 'legendonly'

        scatter = go.Scatter(x=x, y=y, name=name, text=text, mode=mode, hovertemplate=hover_template,
                             showlegend=show_legend, visible=visible, opacity=opacity, **kwargs)
        self.fig.add_trace(scatter, row=row, col=1)
        self.fig.update_traces(xaxis="x1")

    def add_bar_indicator(self, x, y, name: str, row: int, color=None, **kwargs):
        """绘制条形图指标

        绘图API文档：https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html

        函数执行逻辑：

        1. 获取自定义参数 hover_template、show_legend、visible 和 base。这些参数分别对应于鼠标悬停时显示的模板、是否显示图例、是否可见和基线（默认为 True）。
        2. 如果 color 参数为空，则使用 self.color_red 作为颜色。
        3. 使用给定的 x、y 数据创建一个 go.Bar 对象（条形图），并传入以下参数：
            - x: 指标的x轴数据
            - y: 指标的y轴数据
            - marker_line_color: 条形边框的颜色
            - marker_color: 条形填充的颜色
            - name: 指标名称
            - showlegend: 是否显示图例
            - hovertemplate: 鼠标悬停时显示的模板
            - visible: 是否可见
            - base: 基线，默认为 True
        4. 调用 self.fig.add_trace 方法将创建的 go.Bar 对象添加到指定子图中，并更新所有 traces 的 X 轴属性为 "x1"。

        :param x: 指标的x轴
        :param y: 指标的y轴
        :param name: 指标名称
        :param row: 放入第几个子图
        :param color: 指标的颜色，可以是单个颜色，也可以是一个列表，列表长度和y的长度一致，指示每个y的颜色
            比如：color = 'rgba(249,41,62,0.7)' 或者 color = ['rgba(249,41,62,0.7)', 'rgba(0,170,59,0.7)']
        :param kwargs:
        :return:
        """
        hover_template = kwargs.pop('hover_template', '%{y:.3f}')
        show_legend = kwargs.pop('show_legend', True)
        visible = kwargs.pop('visible', True)
        base = kwargs.pop('base', True)
        if color is None:
            color = self.color_red

        bar = go.Bar(x=x, y=y, marker_line_color=color, marker_color=color, name=name,
                     showlegend=show_legend, hovertemplate=hover_template, visible=visible, base=base, **kwargs)
        self.fig.add_trace(bar, row=row, col=1)
        self.fig.update_traces(xaxis="x1")

    def open_in_browser(self, file_name: str = None, **kwargs):
        """在浏览器中打开"""
        if not file_name:
            # file_name = os.path.join(home_path, "kline_chart.html")
            file_name = "kline_chart.html"
        self.fig.update_layout(**kwargs)
        self.fig.write_html(file_name)
        webbrowser.open(file_name)