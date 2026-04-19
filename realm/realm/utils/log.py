"""
logger.py — Centralised coloured logger for REALM.

Usage
-----
    from utils.logger import get_logger, log_info, log_warn, log_error, log_success, logo

    # Module-level (uses default INFO level)
    logger = get_logger(__name__)

    # Entry-point (set level from CLI args)
    logger = get_logger("REALM", level=args.log_level)
    logo()
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------
_RESET    = "\033[0m"
_BOLD     = "\033[1m"
_RED      = "\033[31m"
_GREEN    = "\033[32m"
_YELLOW   = "\033[33m"
_BLUE     = "\033[34m"
_WHITE    = "\033[37m"
_BG_NAVY  = "\033[48;2;14;40;65m"
_FG_WHITE = "\033[97m"

# Only emit colour codes when stdout is a real terminal
_COLOURS_ENABLED: bool = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


# ---------------------------------------------------------------------------
# Custom SUCCESS level (sits between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------
SUCCESS: int = 25
logging.addLevelName(SUCCESS, "SUCCESS")


def _success(self: logging.Logger, msg: object, *args, **kwargs) -> None:  # noqa: ANN001
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, msg, args, **kwargs)


logging.Logger.success = _success  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Coloured formatter
# ---------------------------------------------------------------------------

class _ColoredFormatter(logging.Formatter):
    """
    Applies per-level ANSI colour to log records.

    A new ``logging.Formatter`` is *not* created on every ``format()`` call;
    instead the colour prefix/suffix is prepended/appended directly, which
    avoids the per-call allocation in the original implementation.
    """

    _LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG:   _WHITE,
        logging.INFO:    _BLUE,
        SUCCESS:         _GREEN,
        logging.WARNING: _YELLOW,
        logging.ERROR:   _RED,
        logging.CRITICAL: _RED + _BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not _COLOURS_ENABLED:
            return formatted
        color = self._LEVEL_COLORS.get(record.levelno, _WHITE)
        return f"{color}{formatted}{_RESET}"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(
    name: str = "REALM",
    level: int | str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """
    Return a named logger, creating and configuring it on first call.

    Subsequent calls with the same *name* update the log level and return
    the existing logger — handlers are not duplicated.

    Args:
        name:     Logger name (use ``__name__`` inside library modules).
        level:    Log level as a string ("DEBUG", "INFO", …) or integer
                  (e.g. ``logging.DEBUG``).
        log_file: Optional path to a plain-text log file (no ANSI codes).

    Returns:
        Configured :class:`logging.Logger` instance.

    Raises:
        ValueError: If *level* is not a recognised log-level string or integer.
    """
    if isinstance(level, int):
        numeric_level = level
    else:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(
                f"Invalid log level: {level!r}. "
                "Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )

    logger = logging.getLogger(name)
    logger.propagate = False  # prevent double-printing via root logger

    # Always honour a level change even if handlers already exist
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return logger

    # ---- Console handler -----------------------------------------------
    fmt = "[%(asctime)s] [%(levelname)-7s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        _ColoredFormatter(fmt=fmt, datefmt=datefmt)
    )
    logger.addHandler(console_handler)

    # ---- Optional file handler (plain text, no ANSI) -------------------
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# Modules that want a quick one-liner can do:
#   from utils.logger import log_info
# Entry-point code should call get_logger("REALM", level=args.log_level)
# before using these so the level is set correctly.
# ---------------------------------------------------------------------------

_default_logger = get_logger("REALM")


def log_info(msg: str) -> None:
    _default_logger.info(msg)


def log_warn(msg: str) -> None:
    _default_logger.warning(f"⚠️  {msg}")


def log_error(msg: str) -> None:
    _default_logger.error(f"❌ {msg}")


def log_success(msg: str) -> None:
    _default_logger.success(f"✅ {msg}")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Logo — call explicitly at entry points, never on import
# ---------------------------------------------------------------------------

def logo() -> None:
    """Print the REALM ASCII logo. Call once at your entry point."""
    _logo_text = r"""
           ▄▄▄          
     ▄ ▄  █████  ▄ ▄      ██████  ███████  █████  ██      ███    ███
    ████▄ ██▄██ ▄████     ██   ██ ██      ██   ██ ██      ████  ████
    ▀███████████████▀     ██████  █████   ███████ ██      ██ ████ ██
    ▄██████▀▀▀██████▄     ██   ██ ██      ██   ██ ██      ██  ██  ██
    ███████   ███████     ██   ██ ███████ ██   ██ ███████ ██      ██    
    """
    if _COLOURS_ENABLED:
        _default_logger.info(f"{_BG_NAVY}{_FG_WHITE}{_logo_text}{_RESET}")
    else:
        _default_logger.info(_logo_text)
logo()  # Call at module level to print on first import