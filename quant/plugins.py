"""扩展注册表：策略 / 数据源 / 风控 overlay 的可插拔注册与加载。

三层用法：

1. **内置**：``quant.strategy.library.BUILTIN`` 的九个策略、三种数据源（synthetic / csv / akshare）、
   五个风控 overlay（exposure_cap / max_weight / vol_target / trailing_stop / turnover_limit）。
2. **注册**：在自己的模块里用装饰器注册——

   .. code-block:: python

      from quant.plugins import strategy, source, overlay

      @strategy("my_signal")
      class MySignal(Strategy):
          def generate(self, bars): ...

      @source("mydb")
      def open_mydb(root, **kw):
          return MyDbSource(root, **kw)

      @overlay("hard_cap")
      def hard_cap(limit=0.2):
          def apply(weights, bars, ctx): ...
          return apply

3. **自动加载**：把模块文件放进工程根的 ``plugins/`` 目录（或在 config.toml 的 ``[plugins]`` 段里
   指定 ``files`` / ``dirs``），CLI 启动时会自动 import，注册即可用：``--strategy my_signal``。

注册表只做「名字 → 工厂」的映射，不改变任何既有契约：策略仍然输出目标权重矩阵，
数据源仍然实现 ``load(symbol, start, end)``，overlay 仍然是 ``(weights, bars, ctx) -> weights``。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "PLUGIN_DIR_NAME",
    "build_overlay",
    "build_strategy",
    "describe",
    "load_plugin_dir",
    "load_plugin_file",
    "load_plugins",
    "load_plugins_from_config",
    "load_plugins_from_config_file",
    "open_source",
    "overlay",
    "overlay_names",
    "register_overlay",
    "register_source",
    "register_strategy",
    "source",
    "source_names",
    "strategy",
    "strategy_names",
    "project_root",
]

PLUGIN_DIR_NAME = "plugins"

_STRATEGIES: dict[str, type] = {}
_SOURCES: dict[str, Callable[..., Any]] = {}
_OVERLAYS: dict[str, Callable[..., Any]] = {}

LOADED_FILES: list[str] = []
LOAD_ERRORS: list[tuple[str, str]] = []


def project_root() -> Path:
    """工程根目录（本文件在 <root>/quant/plugins.py）。"""
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- 注册
def register_strategy(name: str, cls: type, *, replace: bool = False) -> type:
    """注册策略类。名字不区分大小写。"""
    key = _key(name)
    if key in _STRATEGIES and not replace:
        raise ValueError(f"策略 {name} 已注册；如需覆盖请传 replace=True")
    _STRATEGIES[key] = cls
    return cls


def register_source(name: str, factory: Callable[..., Any], *, replace: bool = False) -> Callable[..., Any]:
    """注册数据源工厂（返回的对象需实现 ``DataSource.load``）。"""
    key = _key(name)
    if key in _SOURCES and not replace:
        raise ValueError(f"数据源 {name} 已注册；如需覆盖请传 replace=True")
    _SOURCES[key] = factory
    return factory


def register_overlay(name: str, factory: Callable[..., Any], *, replace: bool = False) -> Callable[..., Any]:
    """注册风控 overlay 工厂（返回 ``(weights, bars, ctx) -> weights``）。"""
    key = _key(name)
    if key in _OVERLAYS and not replace:
        raise ValueError(f"overlay {name} 已注册；如需覆盖请传 replace=True")
    _OVERLAYS[key] = factory
    return factory


def strategy(name: str, *, replace: bool = False):
    """装饰器：把类注册为策略。"""

    def deco(cls: type) -> type:
        register_strategy(name, cls, replace=replace)
        return cls

    return deco


def source(name: str, *, replace: bool = False):
    """装饰器：把工厂函数注册为数据源。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        register_source(name, fn, replace=replace)
        return fn

    return deco


def overlay(name: str, *, replace: bool = False):
    """装饰器：把工厂函数注册为风控 overlay。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        register_overlay(name, fn, replace=replace)
        return fn

    return deco


# --------------------------------------------------------------------------- 查询 / 构建
def _key(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("名字不能为空")
    return key


def builtin_strategies() -> dict[str, type]:
    from quant.strategy.library import BUILTIN

    return dict(BUILTIN)


def strategy_names() -> list[str]:
    names = set(builtin_strategies()) | set(_STRATEGIES)
    return sorted(names)


def _accepted_params(target: Any, params: dict[str, Any]) -> dict[str, Any]:
    """只把策略认得的参数传给它。

    ``config.toml`` 的 ``[params]`` 是全局段，不同策略只用其中一部分；
    自定义策略通常不写 ``**kwargs``，直接透传会 TypeError，所以这里先按签名过滤。
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # 内建/ C 实现等取不到签名
        return dict(params)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(params)
    allowed = {
        name
        for name, p in signature.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    dropped = sorted(set(params) - allowed)
    if dropped:
        print(f"[plugins] 策略不认这些参数，已忽略：{', '.join(dropped)}")
    return {key: value for key, value in params.items() if key in allowed}


def build_strategy(name: str, **params: Any):
    """按名字建策略：先查注册表，再查内置库。"""
    key = _key(name)
    if key in _STRATEGIES:
        target = _STRATEGIES[key]
        return target(**_accepted_params(target, params))
    from quant.strategy.library import BUILTIN, build as build_builtin

    target = BUILTIN.get(key)
    if target is None:  # 交给内置库抛它自己的「未知策略」错误
        return build_builtin(key, **params)
    # 按目标类（不是 build 的 **kwargs）过滤参数，配置段里多写的键会被忽略
    return build_builtin(key, **_accepted_params(target, params))


def source_names() -> list[str]:
    return sorted({"synthetic", "csv", "akshare", "web"} | set(_SOURCES))


def open_source(name: str, **kwargs: Any):
    """按名字建数据源。

    内置：``synthetic``（直接透传参数）、``csv``（需要 ``root``，可选 ``symbol_column``）、
    ``akshare``（需要装了 akshare，可选 ``adjust``）、``web``（联网抓，三源互备，
    ``source="auto"`` / ``adjust`` / ``retries`` / ``timeout``）。自定义源直接调用其工厂。
    """
    key = _key(name)
    if key in _SOURCES:
        return _SOURCES[key](**kwargs)
    if key == "synthetic":
        from quant.data import SyntheticSource

        return SyntheticSource(**kwargs)
    if key == "csv":
        from quant.data import CsvSource

        params = dict(kwargs)
        root = params.pop("root", None) or params.pop("path", None) or params.pop("dir", None)
        if root is None:
            raise ValueError("csv 数据源需要 root 参数（CSV 所在目录）")
        return CsvSource(root, **params)
    if key == "akshare":
        from quant.data import AkshareSource

        return AkshareSource(**kwargs)
    if key == "web":
        from quant.data.web_source import WebKlineSource

        return WebKlineSource(**kwargs)
    raise KeyError(f"未知数据源 {name}；可选：{', '.join(source_names())}")


def overlay_names() -> list[str]:
    return sorted({"exposure_cap", "max_weight", "vol_target", "trailing_stop", "turnover_limit"} | set(_OVERLAYS))


def build_overlay(name: str, **kwargs: Any):
    """按名字建单个 overlay。"""
    key = _key(name)
    if key in _OVERLAYS:
        return _OVERLAYS[key](**kwargs)
    from quant import risk

    mapping = {
        "exposure_cap": risk.exposure_cap,
        "max_weight": risk.max_weight,
        "vol_target": risk.vol_target,
        "trailing_stop": risk.trailing_stop,
        "turnover_limit": risk.turnover_limit,
    }
    if key not in mapping:
        raise KeyError(f"未知 overlay {name}；可选：{', '.join(overlay_names())}")
    return mapping[key](**kwargs)


def describe() -> dict[str, Any]:
    """注册表快照，供 ``python -m quant plugins`` 打印。"""
    from quant.strategy.library import BUILTIN

    return {
        "strategies": {
            "builtin": sorted(BUILTIN),
            "custom": sorted(_STRATEGIES),
        },
        "sources": {
            "builtin": ["akshare", "csv", "synthetic"],
            "custom": sorted(_SOURCES),
        },
        "overlays": {
            "builtin": sorted(["exposure_cap", "max_weight", "vol_target", "trailing_stop", "turnover_limit"]),
            "custom": sorted(_OVERLAYS),
        },
        "loadedFiles": list(LOADED_FILES),
        "errors": [{"path": p, "error": e} for p, e in LOAD_ERRORS],
    }


# --------------------------------------------------------------------------- 加载
def load_plugin_file(path: str | Path) -> str:
    """import 一个插件文件；返回模块名。出错时抛异常并记录。"""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"插件文件不存在：{p}")
    if str(p) in LOADED_FILES:
        # 幂等：同一个文件只执行一次，避免二次注册报「已注册」
        return f"cached_{p.stem}"
    mod_name = f"quant_user_plugin_{p.stem}_{abs(hash(str(p))) % 10**8}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载插件：{p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _apply_module_declarations(module, str(p))
    LOADED_FILES.append(str(p))
    return mod_name


def _apply_module_declarations(module: Any, path: str) -> None:
    """支持三种声明方式：装饰器 / 模块级字典 / ``register(api)`` 钩子。"""
    for attr, register in (("STRATEGIES", register_strategy), ("SOURCES", register_source), ("OVERLAYS", register_overlay)):
        table = getattr(module, attr, None)
        if isinstance(table, dict):
            for name, obj in table.items():
                register(name, obj, replace=True)
    hook = getattr(module, "register", None)
    if callable(hook):
        hook(PluginAPI(path))


class PluginAPI:
    """传给插件 ``register(api)`` 钩子的最小 API。"""

    def __init__(self, path: str = "") -> None:
        self.path = path

    def strategy(self, name: str, cls: type) -> type:
        return register_strategy(name, cls, replace=True)

    def source(self, name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
        return register_source(name, factory, replace=True)

    def overlay(self, name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
        return register_overlay(name, factory, replace=True)


def load_plugin_dir(path: str | Path) -> list[str]:
    """import 目录下所有 ``*.py``（跳过 ``_`` 开头的文件）；返回成功加载的文件。"""
    p = Path(path)
    if not p.is_dir():
        return []
    loaded: list[str] = []
    for f in sorted(p.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            load_plugin_file(f)
            loaded.append(str(f))
        except Exception:
            LOAD_ERRORS.append((str(f), traceback.format_exc(limit=3)))
    return loaded


def load_plugins(
    paths: Iterable[str | Path] | None = None,
    *,
    dirs: Iterable[str | Path] = (),
    files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """加载插件文件/目录。

    ``paths`` 里目录与文件混放也可以；``dirs`` / ``files`` 是显式的额外入口。
    单个文件失败不会中断整体加载，错误记进 ``LOAD_ERRORS`` 并由 ``describe()`` 暴露。
    """
    loaded: list[str] = []
    entries: list[Path] = []
    for item in list(paths or []) + list(dirs) + list(files):
        entries.append(Path(item))
    for entry in entries:
        if entry.is_dir():
            loaded.extend(load_plugin_dir(entry))
        elif entry.is_file():
            try:
                load_plugin_file(entry)
                loaded.append(str(entry.resolve()))
            except Exception:
                LOAD_ERRORS.append((str(entry), traceback.format_exc(limit=3)))
    return {"loaded": loaded, "errors": [{"path": p, "error": e} for p, e in LOAD_ERRORS]}


def load_plugins_from_config(raw: dict[str, Any] | None, *, root: str | Path | None = None) -> dict[str, Any]:
    """按 ``[plugins]`` 段加载：``dirs`` / ``files``，默认加上工程根 ``plugins/`` 目录。"""
    root_path = Path(root) if root is not None else project_root()
    section = (raw or {}).get("plugins") or {}
    dirs = [section.get("dirs")] if isinstance(section.get("dirs"), str) else list(section.get("dirs") or [])
    files = [section.get("files")] if isinstance(section.get("files"), str) else list(section.get("files") or [])
    resolved_dirs = [root_path / d for d in dirs]
    resolved_files = [root_path / f for f in files]
    if section.get("autoload", True):
        resolved_dirs.append(root_path / PLUGIN_DIR_NAME)
    return load_plugins(dirs=resolved_dirs, files=resolved_files)


def load_plugins_from_config_file(path: str | Path | None, *, root: str | Path | None = None) -> dict[str, Any]:
    """读 TOML 配置文件里的 ``[plugins]`` 段并加载（文件不存在时只加载默认目录）。"""
    raw: dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.is_file():
            import tomllib

            raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return load_plugins_from_config(raw, root=root)
