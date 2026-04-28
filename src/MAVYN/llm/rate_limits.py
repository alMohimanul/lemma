"""Persistent rate limit tracking for Groq models."""
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


def _store_path() -> Path:
    return Path.home() / ".MAVYN" / "rate_limits.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _next_midnight_utc() -> datetime:
    n = _now()
    return (n + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_retry_seconds(error_msg: str) -> int:
    """Extract 'try again in Xs' from a Groq error message, fallback 60."""
    m = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", error_msg, re.IGNORECASE)
    if not m:
        return 60
    value, unit = float(m.group(1)), m.group(2).lower()
    seconds = value / 1000 if unit == "ms" else value
    return max(1, int(seconds) + 1)


class RateLimitStore:
    """Tracks per-model RPM and RPD blocks, persisted to ~/.MAVYN/rate_limits.json."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _store_path()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return dict(json.loads(self._path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def _get(self, model: str, key: str) -> Optional[datetime]:
        raw = self._data.get(model, {}).get(key)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _set(self, model: str, key: str, value: Optional[datetime]) -> None:
        if model not in self._data:
            self._data[model] = {}
        self._data[model][key] = value.isoformat() if value else None
        self._save()

    def is_available(self, model: str) -> bool:
        now = _now()
        rpm_until = self._get(model, "rpm_blocked_until")
        rpd_until = self._get(model, "rpd_blocked_until")
        if rpm_until and now < rpm_until:
            return False
        if rpd_until and now < rpd_until:
            return False
        return True

    def mark_rpm_limited(self, model: str, error_msg: str = "") -> None:
        seconds = _parse_retry_seconds(error_msg)
        self._set(model, "rpm_blocked_until", _now() + timedelta(seconds=seconds))

    def mark_rpd_limited(self, model: str) -> None:
        self._set(model, "rpd_blocked_until", _next_midnight_utc())

    def cooldown_display(self, model: str) -> str:
        now = _now()
        rpd_until = self._get(model, "rpd_blocked_until")
        rpm_until = self._get(model, "rpm_blocked_until")

        if rpd_until and now < rpd_until:
            remaining = rpd_until - now
            h, rem = divmod(int(remaining.total_seconds()), 3600)
            m = rem // 60
            return f"Daily limit — resets in {h}h {m}m"

        if rpm_until and now < rpm_until:
            secs = max(0, int((rpm_until - now).total_seconds()))
            return f"RPM cooldown — {secs}s"

        return "Available"


def classify_rate_limit(error_msg: str) -> str:
    """Return 'rpd', 'rpm', or 'other' from a Groq 429 error message."""
    lower = error_msg.lower()
    if "per day" in lower or "rpd" in lower or "requests per day" in lower:
        return "rpd"
    if (
        "per minute" in lower
        or "rpm" in lower
        or "requests per minute" in lower
        or "tokens per minute" in lower
    ):
        return "rpm"
    return "rpm"  # conservative default: treat unknown 429 as short cooldown
