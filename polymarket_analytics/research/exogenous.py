"""Exogenous data provider interfaces (no fabricated PIT series)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence


@dataclass(frozen=True)
class ExogenousPoint:
    ts: datetime
    key: str
    value: float
    meta: dict[str, Any] | None = None


class ExogenousProvider(ABC):
    """Point-in-time safe exogenous series provider."""

    name: str = "base"
    available: bool = False
    reason_unavailable: str = "not configured"

    @abstractmethod
    def get_asof(
        self,
        key: str,
        as_of: datetime,
    ) -> ExogenousPoint | None:
        """Return last observation with timestamp <= as_of (strict PIT)."""

    def get_range(
        self,
        key: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[ExogenousPoint]:
        return []


class NullExogenousProvider(ExogenousProvider):
    name = "null"
    available = False
    reason_unavailable = "No PIT exogenous corpus in repo"

    def get_asof(self, key: str, as_of: datetime) -> ExogenousPoint | None:
        return None


class NewsSentimentProvider(NullExogenousProvider):
    name = "news_sentiment"
    reason_unavailable = "No point-in-time news sentiment store"


class OnchainFlowProvider(NullExogenousProvider):
    name = "onchain_flow"
    reason_unavailable = "No indexed wallet/flow lake wired"


class MacroSeriesProvider(NullExogenousProvider):
    name = "macro_series"
    reason_unavailable = "No FRED/macro PIT snapshot table"


class SocialVolumeProvider(NullExogenousProvider):
    name = "social_volume"
    reason_unavailable = "No social firehose with as-of timestamps"


class RelatedMarketGraphProvider(NullExogenousProvider):
    name = "related_market_graph"
    reason_unavailable = "No curated lead-lag / substitute market graph"


class SportsScheduleProvider(NullExogenousProvider):
    name = "sports_schedule"
    reason_unavailable = "No sports fixture/injury PIT feed"


class WeatherProvider(NullExogenousProvider):
    name = "weather"
    reason_unavailable = "No weather forecast archive with issue times"


class PollingProvider(NullExogenousProvider):
    name = "polling"
    reason_unavailable = "No polling aggregate archive"


class ExchangeRefPriceProvider(NullExogenousProvider):
    name = "exchange_ref_price"
    reason_unavailable = "No crypto/ref spot PIT series for crypto markets"


class VolatilityIndexProvider(NullExogenousProvider):
    name = "volatility_index"
    reason_unavailable = "No VIX/IV archive"


DEFAULT_PROVIDERS: tuple[ExogenousProvider, ...] = (
    NewsSentimentProvider(),
    OnchainFlowProvider(),
    MacroSeriesProvider(),
    SocialVolumeProvider(),
    RelatedMarketGraphProvider(),
    SportsScheduleProvider(),
    WeatherProvider(),
    PollingProvider(),
    ExchangeRefPriceProvider(),
    VolatilityIndexProvider(),
)


def provider_status() -> list[dict[str, Any]]:
    return [
        {
            "name": p.name,
            "available": p.available,
            "reason_unavailable": p.reason_unavailable,
        }
        for p in DEFAULT_PROVIDERS
    ]
