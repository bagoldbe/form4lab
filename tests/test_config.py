"""Documents the OSS config surface (see form4lab/config.py).

The sizes, hold days, and vol-target/mass-disappearance fields asserted
below are the actual shipped defaults. Config intentionally does not
define local_database_url or a dual-engine analysis-DB workflow (see
test_database_retry.py).
"""
from form4lab.config import Settings, SecConfig, ScoringConfig, SignalConfig, SchedulerConfig, AlpacaConfig


def test_default_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Must also prevent reading .env file
    settings = Settings(_env_file=None)
    assert "sqlite" in settings.database_url


def test_sec_identity_default(monkeypatch):
    """No working default — required at startup (see form4lab.main's guard,
    which raises RuntimeError when this is empty)."""
    monkeypatch.delenv("SEC_IDENTITY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.sec_identity == ""


def test_env_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    settings = Settings()
    assert settings.database_url == "postgresql://localhost/test"


def test_strategy_path_default():
    """Default strategy resolves to the shipped cluster_buy example."""
    settings = Settings(_env_file=None)
    assert settings.strategy_path == "form4lab.strategies.cluster_buy:ClusterBuyStrategy"


def test_cors_origins_default_empty():
    settings = Settings(_env_file=None)
    assert settings.cors_origins == []


# --- SecConfig ---

def test_sec_config_defaults():
    cfg = SecConfig()
    assert cfg.max_requests_per_second == 9.0
    assert cfg.max_retries == 4
    assert cfg.rate_limit_wait_seconds == 65
    assert cfg.http_timeout_seconds == 30


# --- ScoringConfig ---

def test_scoring_config_defaults():
    cfg = ScoringConfig()
    assert cfg.hit_rate_alpha_0 == 5.5
    assert cfg.hit_rate_beta_0 == 4.5
    assert cfg.hit_rate_baseline == 0.55
    assert cfg.prior_excess_return == 0.01
    assert cfg.shrinkage_k == 5
    assert cfg.default_volatility == 0.15
    assert cfg.elite_skill_min == 1.5
    assert cfg.elite_confidence_min == 0.80
    assert cfg.elite_sample_min == 6
    assert cfg.strong_skill_min == 0.8
    assert cfg.insufficient_sample_min == 3
    # horizon_days selects which outcome columns
    # (hit_60d/excess_return_60d, etc.) scoring reads.
    assert cfg.horizon_days == 60


def test_scoring_config_env_override(monkeypatch):
    monkeypatch.setenv("SCORING_ELITE_SKILL_MIN", "2.0")
    cfg = ScoringConfig()
    assert cfg.elite_skill_min == 2.0


# --- SignalConfig ---

def test_signal_config_defaults():
    cfg = SignalConfig()
    assert cfg.cluster_window_days == 7
    assert cfg.hold_days_default == 60
    assert cfg.conviction_w_skill == 0.4
    assert cfg.conviction_w_role == 0.3
    assert cfg.conviction_w_size == 0.3
    assert cfg.default_role_weight == 0.5
    # Off-by-default knobs:
    assert cfg.drawdown_falling_conviction_boost == 1.0
    assert cfg.preempt_enabled is False
    assert cfg.sell_exit_enabled is False
    assert cfg.stop_loss_pct is None


# --- SchedulerConfig ---

def test_scheduler_config_defaults():
    cfg = SchedulerConfig()
    assert cfg.timezone == "US/Eastern"
    assert cfg.ingest_hour == 21
    assert cfg.outcomes_hour == 20
    assert cfg.scores_hour == 22
    assert cfg.prices_hour == 18
    assert cfg.prices_minute == 30
    assert cfg.spy_rebalance_hour == 9
    assert cfg.spy_rebalance_minute == 26


# --- Settings nesting ---

def test_settings_has_nested_configs():
    settings = Settings()
    assert isinstance(settings.sec, SecConfig)
    assert isinstance(settings.scoring, ScoringConfig)
    assert isinstance(settings.signal, SignalConfig)
    assert isinstance(settings.scheduler, SchedulerConfig)
    assert isinstance(settings.alpaca, AlpacaConfig)


# --- AlpacaConfig ---

def test_alpaca_config_defaults(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ALPACA_ENABLED", raising=False)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    cfg = AlpacaConfig()
    assert cfg.api_key == ""
    assert cfg.secret_key == ""
    assert cfg.paper is True
    assert cfg.enabled is False
    # Role-tiered sizing is available but off by default (both sizes
    # equal; see AlpacaConfig's comment).
    assert cfg.base_size_pct == 0.05
    assert cfg.csuite_size_pct == 0.05
    assert cfg.hold_days == 60
    assert cfg.max_positions_per_insider_ticker == 1
    assert cfg.max_positions_per_ticker == 2
    assert cfg.drawdown_threshold is None
    assert cfg.margin_multiplier == 1.0


def test_alpaca_config_on_settings():
    from form4lab.config import settings
    assert hasattr(settings, "alpaca")
    assert settings.alpaca.paper is True
    # enabled/api_key depend on the environment — just check the attribute exists
    assert isinstance(settings.alpaca.enabled, bool)


def test_alpaca_spy_parking_defaults():
    """SPY parking should be disabled by default with 20% buffer."""
    cfg = AlpacaConfig()
    assert cfg.spy_parking_enabled is False
    assert cfg.spy_parking_buffer == 0.20


def test_alpaca_reconcile_mass_disappearance_limit_default():
    """Circuit breaker: default threshold is 2 simultaneous
    disappearances before the breaker holds everything for review."""
    cfg = AlpacaConfig()
    assert cfg.reconcile_mass_disappearance_limit == 2


def test_alpaca_vol_target_k_default_none():
    """No default risk budget is shipped — vol-targeted sizing is
    unavailable until an operator explicitly sets ALPACA_VOL_TARGET_K."""
    cfg = AlpacaConfig()
    assert cfg.vol_target_k is None
    assert cfg.vol_targeting_enabled is False
