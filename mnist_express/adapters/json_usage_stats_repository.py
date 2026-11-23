"""Adaptateur du port `UsageStatsRepository` : les statistiques sont un fichier JSON."""
import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..core.model import UsageStats

logger = logging.getLogger(__name__)

class JsonUsageStatsRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> UsageStats:
        if not self.path.exists():
            logger.info("Monitoring file does not exist yet: %s", self.path)
            return UsageStats()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return UsageStats(**raw)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(
                "Failed to load monitoring stats from %s: %s | using defaults",
                self.path,
                e,
            )
            return UsageStats()

    def save(self, stats: UsageStats) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(asdict(stats), f, indent=2)
        logger.info("Monitoring stats saved to %s", self.path)