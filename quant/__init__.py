"""A股量化交易系统 —— 自建轻量事件驱动引擎。

模块分层（自下而上）::

    data        行情数据源（CSV / 合成 / akshare）与缓存
    indicators  技术指标（纯 pandas，无 TA-Lib 依赖）
    strategy    策略抽象与示例策略
    backtest    撮合、账户、事件驱动引擎、绩效指标
    risk        风控（仓位上限 / 止损 / 回撤熔断）
    execution   执行适配层（模拟盘；实盘接口留空并显式拒绝）
    report      回测报表（Markdown / 自绘 SVG）

设计约束
--------
1. 只依赖 ``pandas`` 与 ``numpy``；行情数据用 CSV 落盘，离线可复现。
2. 撮合一律发生在**下一根 K 线**（默认次日开盘），杜绝未来函数。
3. A股规则内建：T+1、100 股整手、涨跌停、最低佣金、印花税、过户费。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
