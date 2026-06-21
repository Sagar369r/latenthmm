"""
V7 Master Config — Single source of truth for ALL parameters.
Now proxying values from config/config.yaml for dynamic editing.
All secrets via environment variables. No hardcoded credentials.
"""
from __future__ import annotations
import os
from .config_loader import get_config

_cfg = get_config()

MODELS_DIR          = os.getenv("V7_MODELS_DIR", "models")
CHECKPOINT_SDE      = os.path.join(MODELS_DIR, "sde_best.pth")
CHECKPOINT_EBM      = os.path.join(MODELS_DIR, "ebm_best.pth")
CHECKPOINT_RL       = os.path.join(MODELS_DIR, "rl_best.pth")
CHECKPOINT_DECODER  = os.path.join(MODELS_DIR, "latent_decoder.pth")
CHECKPOINT_REGIME   = os.path.join(MODELS_DIR, "regime_xgb.json")
CHECKPOINT_REGIME_HMM = os.path.join(MODELS_DIR, "regime_hmm.pkl")
CHECKPOINT_RISK     = os.path.join(MODELS_DIR, "risk_xgb.json")
CHECKPOINT_WELFORD  = os.path.join(MODELS_DIR, "welford.npz")

# ─── Equity & Risk ────────────────────────────────────────────────────────────
BACKTEST_INITIAL_EQUITY  = _cfg['initial_equity']
PROP_DAILY_DD_PCT        = 0.05
PROP_MAX_DD_PCT          = 0.10
MAX_DAILY_DRAWDOWN_USD   = _cfg['max_daily_drawdown_usd'] # Left for backward compatibility in offline scripts
DAILY_PROFIT_LOCK_PCT    = _cfg['daily_profit_lock_pct']

# ─── Monte Carlo ──────────────────────────────────────────────────────────────
MC_PATHS                 = _cfg['mc_paths']
MC_TIMEOUT_MS            = _cfg['mc_timeout_ms']
MC_CONFIDENCE            = _cfg['mc_confidence']

# ─── Execution ────────────────────────────────────────────────────────────────
SLIPPAGE_PIPS            = _cfg['slippage_pips']
COMMISSION_PER_LOT       = _cfg.get('commission_per_lot', 3.0)
EXECUTION_LATENCY_MS     = _cfg.get('execution_latency_ms', 100)
MAX_POSITION_FRACTION    = _cfg['max_position_fraction']
HALF_KELLY_FRACTION      = _cfg['half_kelly_fraction']
MAX_OPEN_POSITIONS       = _cfg['max_open_positions']
ATR_SL_MULT              = _cfg['atr_sl_mult']
ATR_TP_MULT              = _cfg['atr_tp_mult']

# ─── EBM ──────────────────────────────────────────────────────────────────────
EBM_ENERGY_THRESHOLD     = _cfg['ebm_energy_threshold']
EBM_MCMC_STEPS           = _cfg['ebm_mcmc_steps']
EBM_STEP_SIZE            = _cfg['ebm_step_size']
EBM_NOISE_SCALE          = _cfg['ebm_noise_scale']
EBM_TARGET_AUC           = _cfg['ebm_target_auc']
EBM_MARGIN               = _cfg['ebm_margin']

# ─── Neural SDE ───────────────────────────────────────────────────────────────
SDE_LATENT_DIM           = _cfg['sde_latent_dim']
SDE_DRIFT_HEADS          = _cfg['sde_drift_heads']
SDE_DRIFT_LAYERS         = _cfg['sde_drift_layers']
SDE_DRIFT_DIM_FF         = _cfg['sde_drift_dim_ff']
SDE_DIFFUSION_HIDDEN     = _cfg['sde_diffusion_hidden']
SDE_SOLVER               = _cfg['sde_solver']
SDE_DT                   = _cfg['sde_dt']
SDE_ADJOINT_METHOD       = _cfg['sde_adjoint_method']
SDE_SPECTRAL_NORM        = _cfg['sde_spectral_norm']

# ─── Ingestion ────────────────────────────────────────────────────────────────
RING_BUFFER_SIZE         = _cfg['ring_buffer_size']
WELFORD_CLIP_SIGMA       = _cfg['welford_clip_sigma']
WELFORD_MIN_STD          = _cfg['welford_min_std']
WELFORD_WARMUP_TICKS     = _cfg['welford_warmup_ticks']

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_DIM: int         = _cfg['embedding_dim']

# ── Feature Vector Dynamic Indices ────────────────────────────────────────────
# We dynamically compute slice offsets to prevent hardcoded index misalignments.
_feat_sizes = {
    'tib': 4,
    'dt_enc': 32,
    'log_ret': 5,
    'vol_ratio': 1,
    'spread': 3,
    'mom': 2,
    'kalman': 1,
    'cusum': 1,
    'regime_oh': 4,
    'sideways': 1,
    'xgb_regime': 4,
    'xgb_risk': 1
}

assert sum(_feat_sizes.values()) <= EMBEDDING_DIM, f"Feature sizes sum ({sum(_feat_sizes.values())}) must not exceed EMBEDDING_DIM ({EMBEDDING_DIM})"

def _get_feat_offset(key: str) -> int:
    offset = 0
    for k, v in _feat_sizes.items():
        if k == key: return offset
        offset += v
    return offset

XGB_REGIME_IDX_START: int  = _get_feat_offset('xgb_regime')
XGB_REGIME_IDX_END: int    = XGB_REGIME_IDX_START + 4
XGB_RISK_IDX: int          = _get_feat_offset('xgb_risk')
CWT_SCALES               = _cfg['cwt_scales']
CWT_WAVELET              = _cfg['cwt_wavelet']
TIB_WINDOW               = _cfg['tib_window']

# ── Dynamic Thresholds ────────────────────────────────────────────────────────
RISK_XGB_THRESHOLD: float       = 0.5
EMBED_WAVELET_HF_THRESHOLD: float = 0.3
SEQUENCE_LENGTH          = _cfg['sequence_length']
MICRO_HORIZON_TICKS      = _cfg['micro_horizon_ticks']
MACRO_HORIZON_TICKS      = _cfg['macro_horizon_ticks']
KALMAN_PROCESS_NOISE     = _cfg['kalman_process_noise']
KALMAN_MEASUREMENT_NOISE = _cfg['kalman_measurement_noise']
CUSUM_THRESHOLD          = _cfg['cusum_threshold']

# ─── SDE Training ─────────────────────────────────────────────────────────────
SDE_EPOCHS               = _cfg['sde_epochs']
SDE_BATCH_SIZE           = _cfg['sde_batch_size']
SDE_LR                   = _cfg['sde_lr']
SDE_LR_SCHEDULER         = _cfg['sde_lr_scheduler']
SDE_GRAD_CLIP            = _cfg['sde_grad_clip']

# ─── EBM Training ─────────────────────────────────────────────────────────────
EBM_EPOCHS               = _cfg['ebm_epochs']
EBM_BATCH_SIZE           = _cfg['ebm_batch_size']
EBM_LR                   = _cfg['ebm_lr']

# ─── RL Training ──────────────────────────────────────────────────────────────
RL_EPISODES              = _cfg['rl_episodes']
RL_GAMMA                 = _cfg['rl_gamma']
RL_LR_ACTOR              = _cfg['rl_lr_actor']
RL_LR_CRITIC             = _cfg['rl_lr_critic']
RL_ENTROPY_COEF          = _cfg['rl_entropy_coef']
RL_DRAWDOWN_PENALTY      = _cfg['rl_drawdown_penalty']
RL_SLIPPAGE_PENALTY      = _cfg['rl_slippage_penalty']
RL_CONTEXT_BUFFER_SIZE   = _cfg['rl_context_buffer_size']

# ─── CPCV ─────────────────────────────────────────────────────────────────────
CPCV_N_SPLITS            = _cfg['cpcv_n_splits']
CPCV_N_TEST_SPLITS       = _cfg['cpcv_n_test_splits']
CPCV_EMBARGO_TICKS       = _cfg['cpcv_embargo_ticks']
CPCV_PURGE_TICKS         = _cfg['cpcv_purge_ticks']

# ─── XGBoost / ML ─────────────────────────────────────────────────────────────
XGB_REGIME_N_ESTIMATORS  = _cfg['xgb_regime_n_estimators']
XGB_REGIME_MAX_DEPTH     = _cfg['xgb_regime_max_depth']
XGB_REGIME_LR            = _cfg['xgb_regime_lr']
XGB_REGIME_SUBSAMPLE     = _cfg['xgb_regime_subsample']
XGB_REGIME_COLSAMPLE     = _cfg['xgb_regime_colsample']
XGB_REGIME_TARGET_ACC    = _cfg['xgb_regime_target_acc']

XGB_RISK_N_ESTIMATORS    = _cfg['xgb_risk_n_estimators']
XGB_RISK_MAX_DEPTH       = _cfg['xgb_risk_max_depth']
XGB_RISK_LR              = _cfg['xgb_risk_lr']
XGB_RISK_TARGET_AUC      = _cfg['xgb_risk_target_auc']
XGB_RISK_DRAWDOWN_THRESH = _cfg['xgb_risk_drawdown_thresh']

VOL_FOREST_N_ESTIMATORS  = _cfg['vol_forest_n_estimators']
VOL_FOREST_MAX_DEPTH     = _cfg['vol_forest_max_depth']
VOL_FOREST_CONTAMINATION = _cfg['vol_forest_contamination']

HYBRID_RISK_OVERRIDE_THRESH = _cfg['hybrid_risk_override_thresh']
HYBRID_ANOMALY_BLOCK        = _cfg['hybrid_anomaly_block']

MC_BARRIER_TIMEOUT          = 2.0  # seconds to wait for Monte Carlo simulation

# ─── Validation Gates ─────────────────────────────────────────────────────────
TARGET_CALMAR_RATIO      = _cfg['target_calmar_ratio']
TARGET_SHARPE            = _cfg['target_sharpe']
TARGET_WIN_RATE          = _cfg['target_win_rate']
TARGET_MAX_DD_USD        = _cfg['target_max_dd_usd']
TARGET_TICK_LATENCY_MS   = _cfg['target_tick_latency_ms']
PROP_PASS_PROBABILITY    = _cfg['prop_pass_probability']

# ─── Dukascopy Data Source ────────────────────────────────────────────────────
DUKASCOPY_HISTORY_MONTHS    = _cfg['dukascopy_history_months']
DUKASCOPY_TICK_URL          = _cfg['dukascopy_tick_url']
DUKASCOPY_LIVE_WS_URL       = os.getenv("DUKASCOPY_WS_URL", "wss://overnode.dukascopy.com/quotestream/v2")
DUKASCOPY_JFOREX_USER       = os.getenv("DUKASCOPY_USER", "")
DUKASCOPY_JFOREX_PASS       = os.getenv("DUKASCOPY_PASS", "")

# ─── Symbols ──────────────────────────────────────────────────────────────────
SYMBOLS                  = _cfg['symbols']
PRIMARY_SYMBOL           = _cfg['primary_symbol']

def get_pip_size(symbol: str) -> float:
    return _cfg['symbol_overrides'].get(symbol, {}).get('pip_size', _cfg['pip_default']['size'])

def get_pip_value(symbol: str) -> float:
    return _cfg['symbol_overrides'].get(symbol, {}).get('pip_value_per_lot', _cfg['pip_default']['value_per_lot'])

# Provide backward-compatible dictionaries by eagerly evaluating the getters
PIP_SIZE: dict[str, float] = {sym: get_pip_size(sym) for sym in SYMBOLS}
PIP_VALUE_PER_LOT: dict[str, float] = {sym: get_pip_value(sym) for sym in SYMBOLS}

# ─── Prop Firm Profiles ───────────────────────────────────────────────────────
PROP_FIRM_PROFILES = _cfg['prop_profiles']

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST                 = _cfg['api_host']
API_PORT                 = _cfg['api_port']

# ─── Hardcoded Math & Logic ───────────────────────────────────────────────────
MATH_NS_CONVERSION       = _cfg['math_ns_conversion']
MATH_MIN_DELTA_T         = _cfg['math_min_delta_t']

LIVE_MIN_BUFFER_TICKS    = _cfg['live_min_buffer_ticks']
LIVE_ATR_LOOKBACK_TICKS  = _cfg['live_atr_lookback_ticks']
LIVE_ATR_CHUNKS          = _cfg['live_atr_chunks']
LIVE_ATR_CHUNK_SIZE      = _cfg['live_atr_chunk_size']
LIVE_ATR_MIN             = _cfg['live_atr_min']
LIVE_FEATURE_N_RECENT    = _cfg['live_feature_n_recent']
LIVE_MC_THROTTLE_SEC     = _cfg['live_mc_throttle_sec']
LIVE_MC_INTEGRATION_STEPS= _cfg['live_mc_integration_steps']
LIVE_HEARTBEAT_SLEEP_SEC = _cfg['live_heartbeat_sleep_sec']

TRAIN_DEFAULT_MONTHS     = _cfg['train_default_months']
TRAIN_STRIDE             = _cfg['train_stride']
TRAIN_STRIDE_DRY         = _cfg['train_stride_dry']
TRAIN_SEQUENCE_LENGTH    = _cfg['train_sequence_length']
TRAIN_VAL_SPLIT          = _cfg['train_val_split']
TRAIN_SDE_EPOCHS_DRY     = _cfg['train_sde_epochs_dry']
TRAIN_DECODER_BATCH_SIZE = _cfg['train_decoder_batch_size']
TRAIN_DECODER_EPOCHS     = _cfg['train_decoder_epochs']
TRAIN_DECODER_EPOCHS_DRY = _cfg['train_decoder_epochs_dry']
TRAIN_RL_EPISODES        = _cfg['train_rl_episodes']
TRAIN_RL_EPISODES_DRY    = _cfg['train_rl_episodes_dry']

VOL_Z_WINDOW             = _cfg['vol_z_window']
VOL_MIN_PERIODS          = _cfg['vol_min_periods']
VOL_CLIP_LOWER           = _cfg['vol_clip_lower']
VOL_Z_CLIP               = _cfg['vol_z_clip']
VOL_TIB_EMA_SPAN         = _cfg['vol_tib_ema_span']
VOL_SHORT_WINDOW         = _cfg['vol_short_window']
VOL_LONG_WINDOW          = _cfg['vol_long_window']
VOL_OFI_SPAN_1           = _cfg['vol_ofi_span_1']
VOL_OFI_SPAN_2           = _cfg['vol_ofi_span_2']

# ─── Extreme Hardcode Elimination Phase 2 ─────────────────────────────────────
EMBED_XGB_CACHE_TICKS       = _cfg['embed_xgb_cache_ticks']
EMBED_KALMAN_VEL_MIN_LEN    = _cfg['embed_kalman_vel_min_len']
EMBED_COMPUTE_N_RECENT      = _cfg['embed_compute_n_recent']
EMBED_MIN_BIDS_LEN          = _cfg['embed_min_bids_len']
EMBED_STALE_TIME_SEC        = _cfg['embed_stale_time_sec']
EMBED_STALE_TIME_NORM       = _cfg['embed_stale_time_norm']
EMBED_MIN_CLIP_12           = _cfg['embed_min_clip_12']
EMBED_MIN_CLIP_8            = _cfg['embed_min_clip_8']
EMBED_MIN_CLIP_5            = _cfg['embed_min_clip_5']
EMBED_MIN_CLIP_4            = _cfg['embed_min_clip_4']
EMBED_LOG_RET_SLICE         = _cfg['embed_log_ret_slice']
EMBED_VAR_CLIP              = _cfg['embed_var_clip']
EMBED_SPREAD_Z_CLIP         = _cfg['embed_spread_z_clip']
EMBED_SPREAD_MA_CLIP        = _cfg['embed_spread_ma_clip']
EMBED_MOM_5                 = _cfg['embed_mom_5']
EMBED_MOM_20                = _cfg['embed_mom_20']
EMBED_HF_ENERGY_SLICE       = _cfg['embed_hf_energy_slice']
EMBED_COL_KALMAN            = _get_feat_offset('kalman')
EMBED_COL_CUSUM             = _get_feat_offset('cusum')
EMBED_COL_WAVELET           = _get_feat_offset('dt_enc') + 4  # scale 4
EMBED_REGIME_ROLLING_VAR    = _cfg['embed_regime_rolling_var']

RISK_PROP_SIM_DAYS          = _cfg['risk_prop_sim_days']
RISK_PROP_SIM_SEED          = _cfg['risk_prop_sim_seed']
RISK_PROP_SIM_WORKERS       = _cfg['risk_prop_sim_workers']
RISK_Z_95                   = _cfg['risk_z_95']
RISK_PROP_SIM_DD_LIMIT      = _cfg['risk_prop_sim_drawdown_limit']
RISK_TAIL_TRADING_DAYS      = _cfg['risk_tail_trading_days']
RISK_TAIL_VAR_99            = _cfg['risk_tail_var_99']
RISK_TAIL_VAR_95            = _cfg['risk_tail_var_95']
RISK_TAIL_VAR_05            = _cfg['risk_tail_var_05']
RISK_TAIL_MIN_STD           = _cfg['risk_tail_min_std']

RISK_MC_TIMESTEPS           = _cfg['risk_mc_timesteps']
RISK_MC_SIMS_INTERNAL       = _cfg['risk_mc_sims_internal']
RISK_MC_PENALTY_MULT        = _cfg['risk_mc_penalty_mult']
RISK_MC_THRESH_BLOCK        = _cfg['risk_mc_threshold_block']
RISK_MC_DIV_PROTECTOR       = _cfg['risk_mc_div_protector']
RISK_MC_MARGIN_NORM         = _cfg['risk_mc_margin_norm']
RISK_MC_THROTTLE_SEC        = _cfg['risk_mc_throttle_sec']
RISK_MC_BARRIER_BLOCKS      = _cfg['risk_mc_barrier_blocks']

RISK_XGB_THRESHOLD          = _cfg['risk_xgb_threshold']
RISK_XGB_BLOCK_THRESHOLD    = _cfg['risk_xgb_block_threshold']
RISK_XGB_DRAWDOWN_LIMIT     = _cfg['risk_xgb_drawdown_limit']
RISK_XGB_ROLLING_VAR        = _cfg['risk_xgb_rolling_var']

INGEST_PRICE_NORM           = _cfg['ingest_price_norm']
INGEST_TIME_NORM_US         = _cfg['ingest_time_norm_us']
INGEST_TIME_NORM_NS         = _cfg['ingest_time_norm_ns']
INGEST_RETRY_BOUNDS         = _cfg['ingest_retry_bounds']
INGEST_TIMEOUT_LOCKS        = _cfg['ingest_timeout_locks']
INGEST_CHUNK_LIMITS         = _cfg['ingest_chunk_limits']
INGEST_HISTORY_MONTHS       = _cfg['ingest_history_months']
INGEST_HISTORY_DAYS_MARGIN  = _cfg['ingest_history_days_margin']
INGEST_PRICE_NORM_SMALL     = _cfg['ingest_price_norm_small']
INGEST_TIME_NORM_MS         = _cfg['ingest_time_norm_ms']
