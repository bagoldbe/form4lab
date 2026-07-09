import logging

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()  # make .env vars available to all nested configs

_logger = logging.getLogger(__name__)


class SecConfig(BaseSettings):
    """SEC EDGAR API configuration."""

    # Rate limiting — SEC hard limit is 10 req/s with 10-min cooldown on violation
    max_requests_per_second: float = 9.0  # stay under 10 req/s limit
    max_retries: int = 4
    rate_limit_wait_seconds: int = 65  # enough for rate to drop below threshold
    http_timeout_seconds: int = 30
    max_connections: int = 2

    model_config = {"env_prefix": "SEC_", "extra": "ignore"}


class ScoringConfig(BaseSettings):
    """Insider credibility scoring hyperparameters."""

    # Bayesian hit rate prior — Beta(5.5, 4.5) = 10 pseudo-obs at 55% hit rate.
    # Beta prior ~55% hit rate per the insider-trading literature.
    hit_rate_alpha_0: float = 5.5
    hit_rate_beta_0: float = 4.5
    hit_rate_baseline: float = 0.55

    # Shrinkage excess return — Normal-Normal model
    prior_excess_return: float = 0.01  # 1% mild positive edge prior
    shrinkage_k: float = 5  # pseudo-observations for prior strength
    default_volatility: float = 0.15  # assumed stdev when <3 observations

    # Credibility weight
    credibility_n0: int = 8  # sample size for full credibility

    # Skill score
    sigma_typical: float = 0.15  # normalizes excess return magnitude
    skill_w_hit_rate: float = 0.5
    skill_w_magnitude: float = 0.5

    # Tier thresholds — illustrative defaults; calibrate to your own data
    elite_skill_min: float = 1.5
    elite_confidence_min: float = 0.80
    elite_sample_min: int = 6
    strong_skill_min: float = 0.8
    strong_confidence_min: float = 0.65
    insufficient_sample_min: int = 3

    # Momentum adjustment — minimum paired data points for regression
    momentum_min_pairs: int = 5

    # Horizon (trading days) used to select outcome columns when scoring —
    # horizon_days=60 maps to the 60d outcome columns; 20/120 selectable.
    horizon_days: int = 60

    model_config = {"env_prefix": "SCORING_", "extra": "ignore"}


class SignalConfig(BaseSettings):
    """Signal generation and portfolio configuration."""

    # Conviction score weights
    conviction_w_skill: float = 0.4
    conviction_w_role: float = 0.3
    conviction_w_size: float = 0.3

    # Cluster detection — days on either side of transaction
    cluster_window_days: int = 7

    # Hold period (trading days) — generic medium-term convention (matches the
    # 60d outcome column). Per-signal hold periods come from the strategy's
    # SignalType declarations.
    hold_days_default: int = 60

    # Drawdown momentum conviction boost — 1.0 = off
    drawdown_falling_conviction_boost: float = 1.0

    # Preemption — optionally sell an underwater position to fund a new signal
    preempt_enabled: bool = False
    preempt_max_per_signal: int = 2  # max positions to sell per new signal
    preempt_loss_threshold: float = -0.15  # only preempt positions worse than this

    # Sell signal early exit — close positions when credible insiders sell
    sell_exit_enabled: bool = False
    sell_cluster_exit_delay: int = 0   # trading days after alert (0 = same day)
    sell_large_exit_delay: int = 5     # trading days after alert

    # Stop loss — disabled by default; position sizing caps per-position risk.
    stop_loss_pct: float | None = None

    # Default role weight for unknown roles
    default_role_weight: float = 0.5

    model_config = {"env_prefix": "SIGNAL_", "extra": "ignore"}


class SchedulerConfig(BaseSettings):
    """Background job schedule configuration."""

    timezone: str = "US/Eastern"
    # SEC rebuilds /submissions/CIK{cik}.json around 8pm ET; run after that
    # so same-day afternoon filings are in the cache when we fetch it.
    ingest_hour: int = 21  # 9pm ET
    ingest_minute: int = 0
    outcomes_hour: int = 20  # 8pm ET
    outcomes_minute: int = 0
    scores_hour: int = 22  # 10pm ET — kept after ingest to avoid cron collision
    scores_minute: int = 0
    prices_hour: int = 18  # 6:30pm ET — before ingest
    prices_minute: int = 30
    misfire_grace_time: int = 3600
    # Continuous ingestion
    continuous_ingestion_interval_seconds: int = 12
    continuous_ingestion_start_hour: int = 9   # 9:00 AM ET
    continuous_ingestion_end_hour: int = 20     # 8:00 PM ET
    # Exit checks
    exits_hour: int = 9
    exits_minute: int = 25  # 9:25 AM ET — 5 min before market open
    # Order sync
    sync_interval_minutes: int = 5
    # SPY parking rebalance
    spy_rebalance_hour: int = 9
    spy_rebalance_minute: int = 26  # 9:26 AM ET — right after exits close at 9:25

    model_config = {"env_prefix": "SCHEDULER_", "extra": "ignore"}


class AlpacaConfig(BaseSettings):
    """Alpaca paper trading configuration."""

    api_key: str = ""
    secret_key: str = ""
    paper: bool = True  # safety: default to paper trading
    enabled: bool = False  # must explicitly opt-in
    # Role-tiered sizing available — set different values to enable.
    base_size_pct: float = 0.05  # non-C-suite position size (% of portfolio)
    csuite_size_pct: float = 0.05  # C-suite position size
    hold_days: int = 60  # trading days before auto-close
    max_positions_per_insider_ticker: int = 1
    max_positions_per_ticker: int = 2
    stop_loss_pct: float | None = None  # disabled; use ALPACA_STOP_LOSS_PCT env var to enable
    drawdown_threshold: float | None = None  # min drawdown from 52wk high to trade; None=disabled
    margin_multiplier: float = 1.0  # leverage cap (1.0=cash only, 1.5=50% margin)
    spy_parking_enabled: bool = False  # park idle cash in SPY; ALPACA_SPY_PARKING_ENABLED
    spy_parking_buffer: float = 0.20   # keep 20% of equity as cash; ALPACA_SPY_PARKING_BUFFER
    # Reconciliation circuit breaker: if more than this many symbols disappear from
    # Alpaca in a single pass, hold ALL of them for review instead of classifying —
    # a simultaneous multi-position disappearance is a platform glitch until proven
    # otherwise.
    reconcile_mass_disappearance_limit: int = 2
    # Volatility-targeted sizing: pct = clamp(k/realized_vol, min, max).
    # OFF by default — sizing falls back to the role-tiered percentages above.
    vol_targeting_enabled: bool = False  # ALPACA_VOL_TARGETING_ENABLED
    vol_targeting_shadow: bool = False   # log the vol-target size but still trade role-tiered
    # No default — vol targeting requires you to choose a risk budget. When None,
    # vol targeting is unavailable and sizing uses the role-tiered fallback path.
    vol_target_k: float | None = None
    vol_target_min_pct: float = 0.03     # floor on position size
    vol_target_max_pct: float = 0.20     # per-position cap on position size
    vol_target_max_ticker_pct: float | None = None  # AGGREGATE per-ticker cap (None=off)
    vol_target_window: int = 20          # trailing trading days for realized vol

    model_config = {"env_prefix": "ALPACA_", "extra": "ignore"}


class Settings(BaseSettings):
    database_url: str = "sqlite:///form4lab.db"
    # Required at startup (see form4lab.main): your name/org + contact email,
    # per SEC EDGAR's fair-access policy.
    sec_identity: str = ""
    strategy_path: str = "form4lab.strategies.cluster_buy:ClusterBuyStrategy"
    scheduler_enabled: bool = True  # SCHEDULER_ENABLED — set false in the web container when a separate scheduler service runs the jobs
    polling_interval_minutes: int = 15
    backfill_years: int = 10
    # Extra CORS origins for the API/dashboard (env CORS_ORIGINS, JSON list).
    cors_origins: list[str] = []

    # Nested configs
    sec: SecConfig = SecConfig()
    scoring: ScoringConfig = ScoringConfig()
    signal: SignalConfig = SignalConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    alpaca: AlpacaConfig = AlpacaConfig()

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
