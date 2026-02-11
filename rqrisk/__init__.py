from .risk import Risk
from .utils import DAILY, WEEKLY, MONTHLY, YEARLY, NATURAL_DAILY

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("rqrisk")
except PackageNotFoundError:
    # 开发模式下，如果包未安装
    __version__ = "0.0.0.dev0"

__all__ = [
    "Risk",
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "YEARLY",
    "NATURAL_DAILY"
]
