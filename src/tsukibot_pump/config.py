"""Config: `.env` (pydantic-settings) for secrets + ops, YAML for thresholds.

Keeping the two surfaces separate lets the user check `config.yaml` into
version control if they want, while `.env` stays gitignored. `Settings` is
loaded eagerly at startup; `Config` is loaded from disk and validated by
Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── .env / environment ─────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Operational settings loaded from environment / .env file.

    Names are deliberately namespaced (`TSUKI_PUMP_*`) so this bot's env vars
    don't collide with the sibling tsuki-edge-bot Polymarket repo when both
    are running on the same machine.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime mode. `paper-mock` runs entirely without a Solana RPC by
    # replaying a recorded firehose; `paper` reads the live chain but never
    # places real orders.
    tsuki_pump_mode: Literal["paper-mock", "paper", "devnet", "mainnet"] = "paper"
    tsuki_pump_config: Path = Path("config.yaml")
    tsuki_pump_state_dir: Path = Path("./state")
    tsuki_pump_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Solana RPC endpoints.
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    solana_devnet_rpc_url: str = "https://api.devnet.solana.com"
    solana_grpc_url: str = ""
    solana_grpc_token: str = ""

    # Hot wallet secret (only required for devnet / mainnet modes).
    solana_hot_wallet_secret: str = ""

    # Telegram.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Optional integrations.
    birdeye_api_key: str = ""
    dexscreener_api_key: str = ""

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def is_live_chain(self) -> bool:
        """True if we'll touch real or devnet chain state. `paper-mock` doesn't."""
        return self.tsuki_pump_mode in {"paper", "devnet", "mainnet"}

    @property
    def places_real_orders(self) -> bool:
        return self.tsuki_pump_mode in {"devnet", "mainnet"}

    @property
    def risks_real_money(self) -> bool:
        return self.tsuki_pump_mode == "mainnet"

    @property
    def effective_rpc_url(self) -> str:
        if self.tsuki_pump_mode == "devnet":
            return self.solana_devnet_rpc_url
        return self.solana_rpc_url


# ── YAML config (strategy thresholds) ──────────────────────────────────────


class BankrollConfig(BaseModel):
    total_sol: float = Field(gt=0)
    single_token_cap_fraction: float = Field(gt=0, le=1)
    daily_drawdown_kill: float = Field(gt=0, le=1)
    total_drawdown_kill: float = Field(gt=0, le=1)
    max_open_positions: int = Field(ge=1)


class SizingConfig(BaseModel):
    fraction_of_kelly: float = Field(gt=0, le=1)
    min_expected_roi: float = Field(ge=0)
    hard_cap_per_trade_sol: float = Field(gt=0)


class ScoringWeights(BaseModel):
    dev_blacklist: float = Field(ge=0, le=1)
    bundle_cluster: float = Field(ge=0, le=1)
    first_kol_touch: float = Field(ge=0, le=1)
    convergence: float = Field(ge=0, le=1)
    curve_graduation: float = Field(ge=0, le=1)
    cto_revival: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoringWeights:
        total = (
            self.dev_blacklist
            + self.bundle_cluster
            + self.first_kol_touch
            + self.convergence
            + self.curve_graduation
            + self.cto_revival
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring.weights must sum to 1.0 (got {total:.6f})")
        return self


class ScoringConfig(BaseModel):
    enter_threshold: float = Field(ge=0, le=100)
    enter_threshold_watchtower_log: float = Field(ge=0, le=100)
    weights: ScoringWeights


class DevBlacklistConfig(BaseModel):
    enabled: bool
    reject_if_known_rugger: bool
    reject_if_dev_token_count_24h_gte: int = Field(ge=1)
    reject_if_dev_token_count_7d_gte: int = Field(ge=1)
    blacklist_csv_path: Path


class BundleClusterConfig(BaseModel):
    enabled: bool
    reject_cluster_concentration_gte: float = Field(gt=0, le=1)
    first_n_buyers: int = Field(ge=2)
    bundle_window_slots: int = Field(ge=1)


class FirstKolTouchConfig(BaseModel):
    enabled: bool
    first_n_buyers: int = Field(ge=1)
    kol_csv_path: Path
    score_per_touch: float = Field(ge=0)
    max_kol_touches: int = Field(ge=1)


class ConvergenceConfig(BaseModel):
    enabled: bool
    window_seconds: int = Field(ge=1)
    min_distinct_kols: int = Field(ge=2)
    score_at_minimum: float = Field(ge=0, le=100)
    score_per_extra_kol: float = Field(ge=0, le=100)


class CurveGraduationConfig(BaseModel):
    enabled: bool
    enter_after_sol_in_curve_gte: float = Field(ge=0)
    min_velocity_sol_per_min: float = Field(ge=0)
    min_distinct_buyers_60s: int = Field(ge=1)


class CtoRevivalConfig(BaseModel):
    enabled: bool
    dev_silent_min_days: int = Field(ge=1)
    min_unique_buyers_24h: int = Field(ge=1)
    min_days_since_launch: int = Field(ge=1)


class FiltersConfig(BaseModel):
    dev_blacklist: DevBlacklistConfig
    bundle_cluster: BundleClusterConfig
    first_kol_touch: FirstKolTouchConfig
    convergence: ConvergenceConfig
    curve_graduation: CurveGraduationConfig
    cto_revival: CtoRevivalConfig


class PaperRealismConfig(BaseModel):
    """Tunables that make paper fills resemble live fills.

    Defaults are calibrated against publicly documented pump.fun trade fees
    (1% per side as of 2025-2026) and Helius / QuickNode latency studies. The
    point is *not* perfect calibration — it's making paper P&L pessimistic
    enough that mainnet doesn't surprise the user later.
    """

    enabled: bool = False
    # Pump.fun protocol fee (bps of notional, applied on every buy and sell).
    pump_fee_bps: float = Field(default=100.0, ge=0, le=10_000)
    # Independent priority fee (in lamports of SOL, added to buy cost / deducted
    # from sell proceeds — separate from the priority_fee_micro_lamports knob
    # used by the live executor planner).
    priority_fee_lamports_p50: int = Field(default=50_000, ge=0)
    priority_fee_lamports_p99: int = Field(default=300_000, ge=0)
    # End-to-end latency (detection → landed slot), in milliseconds. Used to
    # model curve drift between when we observed the price and when our tx
    # would have landed.
    end_to_end_latency_ms_p50: float = Field(default=1_500.0, ge=0)
    end_to_end_latency_ms_p99: float = Field(default=5_000.0, ge=0)
    # Probability that a paper buy / sell "fails" (slippage tolerance exceeded
    # or blockhash expired). Modelled as a Bernoulli draw per attempt.
    buy_fail_prob: float = Field(default=0.10, ge=0, le=1)
    sell_fail_prob: float = Field(default=0.05, ge=0, le=1)
    # Optional deterministic seed for tests / reproducible paper runs. None
    # means "use process-default RNG".
    rng_seed: int | None = None


class ExecutionConfig(BaseModel):
    paper_slippage_bps: float = Field(ge=0)
    priority_fee_micro_lamports: int = Field(ge=0)
    cu_limit: int = Field(ge=10_000)
    paper_realism: PaperRealismConfig = Field(default_factory=PaperRealismConfig)


class ExitLadderStep(BaseModel):
    roi: float = Field(ge=0)
    sell_fraction: float = Field(gt=0, le=1)


class ExitsConfig(BaseModel):
    ladder: list[ExitLadderStep]
    trailing_stop_pct: float = Field(gt=0, le=1)
    dev_drain_exit_fraction: float = Field(gt=0, le=1)
    hard_stop_loss_pct: float = Field(gt=0, le=1)

    @field_validator("ladder")
    @classmethod
    def _ladder_monotonic(cls, v: list[ExitLadderStep]) -> list[ExitLadderStep]:
        if not v:
            raise ValueError("exits.ladder must contain at least one step")
        prev_roi = -1.0
        total_frac = 0.0
        for step in v:
            if step.roi <= prev_roi:
                raise ValueError("exits.ladder must be sorted by ascending roi")
            prev_roi = step.roi
            total_frac += step.sell_fraction
        if total_frac > 1.0 + 1e-6:
            raise ValueError(f"exits.ladder sell_fractions sum to {total_frac:.4f} > 1.0")
        return v


class WatchConfig(BaseModel):
    use_grpc_if_available: bool
    http_poll_interval_seconds: float = Field(gt=0)
    max_token_age_seconds_on_first_sight: float = Field(gt=0)


class NetworkConfig(BaseModel):
    http_timeout_seconds: float = Field(gt=0)
    ws_ping_interval_seconds: float = Field(gt=0)
    ws_reconnect_max_backoff_seconds: float = Field(gt=0)
    rpc_requests_per_second: float = Field(gt=0)


class TelegramConfig(BaseModel):
    notify_on_entry: bool
    notify_on_exit: bool
    notify_on_kill_switch: bool
    notify_on_anomaly: bool
    notify_on_high_score: bool
    daily_digest: bool
    daily_digest_utc_hour: int = Field(ge=0, le=23)


class DashboardConfig(BaseModel):
    refresh_hz: int = Field(ge=1, le=60)


class EventStoreConfig(BaseModel):
    sqlite_path: Path
    jsonl_audit_path: Path


class Config(BaseModel):
    """Root YAML config schema."""

    bankroll: BankrollConfig
    sizing: SizingConfig
    scoring: ScoringConfig
    filters: FiltersConfig
    execution: ExecutionConfig
    exits: ExitsConfig
    watch: WatchConfig
    network: NetworkConfig
    telegram: TelegramConfig
    dashboard: DashboardConfig
    event_store: EventStoreConfig


def load_config(path: str | Path) -> Config:
    """Load and validate the YAML config.

    Accepts both `str` (from argparse) and `Path` (from `Settings`) so call
    sites don't have to coerce.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Copy config.example.yaml to {path} and edit thresholds."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must be a YAML mapping at the top level")
    return Config.model_validate(raw)
