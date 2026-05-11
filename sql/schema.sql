create extension if not exists pgcrypto;

-- =====================================================================
-- Global bot control state. Exactly one row: id = 'global'.
-- =====================================================================
create table if not exists control_state (
    id text primary key default 'global',

    trading_enabled boolean not null default false,
    kill_switch_active boolean not null default false,

    allow_new_entries boolean not null default true,
    allow_ioc_orders boolean not null default true,
    allow_passive_orders boolean not null default true,
    allow_burst_mode boolean not null default true,

    -- normal / conservative / paused / killed / shadow
    mode text not null default 'normal',

    -- Decimal fraction, e.g. 0.15 = 15%.
    max_market_exposure_pct numeric not null default 0.15,

    -- Shadow mode means compute/log intended orders, but do not send to Kalshi.
    shadow_mode_enabled boolean not null default true,

    updated_at timestamptz not null default now(),
    updated_by text,
    reason text,

    constraint control_state_singleton check (id = 'global'),
    constraint control_state_mode_check check (mode in ('normal', 'conservative', 'paused', 'killed', 'shadow')),
    constraint control_state_exposure_check check (max_market_exposure_pct >= 0 and max_market_exposure_pct <= 1)
);

insert into control_state (id)
values ('global')
on conflict (id) do nothing;

-- =====================================================================
-- Per-market/manual overrides. One row per game if an override exists.
-- =====================================================================
create table if not exists market_controls (
    game_id text primary key,

    -- normal / paused / cancelled / blocked / force_conservative
    market_status text not null default 'normal',

    pause_active boolean not null default false,
    cancel_entry boolean not null default false,
    block_new_entries boolean not null default false,
    force_conservative boolean not null default false,

    updated_at timestamptz not null default now(),
    updated_by text,
    reason text,

    constraint market_controls_status_check check (market_status in ('normal', 'paused', 'cancelled', 'blocked', 'force_conservative'))
);

-- =====================================================================
-- Immutable command audit log.
-- Every dashboard/SMS command inserts here before or as it is applied.
-- =====================================================================
create table if not exists control_commands (
    command_id uuid primary key default gen_random_uuid(),

    command_type text not null,
    scope text not null, -- global / market
    game_id text,
    payload jsonb not null default '{}'::jsonb,

    requested_by text not null,
    requested_via text not null, -- streamlit_dashboard / sms / worker_internal
    auth_status text not null,  -- authorized / unauthorized / failed_pin / allowlist_failed

    received_at timestamptz not null default now(),
    applied_at timestamptz,

    status text not null, -- received / applied / rejected / failed
    reason text,

    constraint control_commands_scope_check check (scope in ('global', 'market')),
    constraint control_commands_status_check check (status in ('received', 'applied', 'rejected', 'failed'))
);

-- =====================================================================
-- Live market snapshots: one current row per live game/canonical exposure.
-- The Python worker upserts this every loop.
-- =====================================================================
create table if not exists live_market_snapshots (
    game_id text primary key,
    updated_at timestamptz not null default now(),

    -- Game identity
    home_team text,
    away_team text,
    selected_team text,
    opponent_team text,
    tipoff_ts timestamptz,
    time_to_tipoff_minutes numeric,

    -- Phase/status
    phase text, -- MONITOR / PASSIVE_ELIGIBLE / ACTIVE / PRE_QUALIFIED_ONLY / ENDED / BLOCKED / KILLED
    trading_status text, -- eligible / no_edge / paused / cancelled / blocked / killed / expansion_gate / late_only / stale_data
    expansion_gate_status text,
    first_qualified_lead_hours numeric,

    -- Forecast/market odds
    model_prob numeric,
    model_prob_t20 numeric,
    model_prob_latest_pre_t8 numeric,
    model_prob_change_t20_to_t8 numeric,
    model_prob_changed_t20_to_t8 boolean not null default false,
    model_prob_last_refresh_at timestamptz,
    model_probability_update_count integer,
    market_prob numeric,
    abs_edge numeric,
    norm_edge numeric,
    q_max_price numeric,
    q_exec_all_in_price numeric,

    -- Position target hierarchy
    bankroll_for_sizing_dollars numeric,
    available_cash_after_buffer_dollars numeric,
    half_kelly_target_dollars numeric,
    portfolio_cap_dollars numeric,
    cash_cap_dollars numeric,
    target_position_now_dollars numeric,

    -- Position state
    filled_position_dollars numeric,
    filled_contracts integer,
    reserved_open_order_dollars numeric,
    remaining_position_dollars numeric,

    -- Liquidity/execution caps
    visible_depth_cap_dollars numeric,
    recent_volume_cap_dollars numeric,
    cold_start_cap_dollars numeric,
    rolling_liquidity_cap_dollars numeric,
    cumulative_cap_remaining_dollars numeric,
    allowed_to_try_now_dollars numeric,
    next_child_order_dollars numeric,

    -- Cash-scarcity priority
    cash_limited_mode boolean not null default false,
    cash_priority_rule text,
    cash_priority_rank integer,
    cash_priority_rank_total integer,
    cash_priority_score numeric,
    expected_log_growth_next_child numeric,
    cash_priority_candidate_child_dollars numeric,
    cash_priority_allocated_child_dollars numeric,
    skipped_due_to_cash boolean not null default false,
    q_current_position numeric,
    q_avg_after_child numeric,

    -- Binders
    target_position_binder text, -- half_kelly / portfolio_cap / cash_cap
    execution_binder text, -- visible_depth / recent_volume / cold_start / cumulative / child_cap / remaining_position / none

    -- Activity summary
    number_of_fills integer not null default 0,
    number_of_order_attempts integer not null default 0,
    number_of_trades_made integer not null default 0,
    average_fill_price numeric,
    vwap_lead_hours numeric,
    last_action text,
    last_reject_reason text,
    last_fill_ts timestamptz,
    last_order_ts timestamptz,

    -- Diagnostics
    market_data_ts timestamptz,
    model_snapshot_ts timestamptz,
    injury_data_ts timestamptz,
    orderbook_ts timestamptz,

    constraint live_market_price_checks check (
        (model_prob is null or (model_prob >= 0 and model_prob <= 1)) and
        (model_prob_t20 is null or (model_prob_t20 >= 0 and model_prob_t20 <= 1)) and
        (model_prob_latest_pre_t8 is null or (model_prob_latest_pre_t8 >= 0 and model_prob_latest_pre_t8 <= 1)) and
        (market_prob is null or (market_prob >= 0 and market_prob <= 1)) and
        (q_max_price is null or (q_max_price >= 0 and q_max_price <= 1)) and
        (q_exec_all_in_price is null or (q_exec_all_in_price >= 0 and q_exec_all_in_price <= 1)) and
        (q_current_position is null or (q_current_position >= 0 and q_current_position <= 1)) and
        (q_avg_after_child is null or (q_avg_after_child >= 0 and q_avg_after_child <= 1))
    )
);

-- =====================================================================
-- Route snapshots: route-level details for smart routing.
-- Upsert by (game_id, route_name, market_ticker, outcome_side).
-- =====================================================================
create table if not exists route_snapshots (
    id uuid primary key default gen_random_uuid(),
    game_id text not null,

    -- BUY_YES_SELECTED / BUY_NO_OPPONENT
    route_name text not null,
    market_ticker text not null,
    outcome_side text not null, -- yes / no

    q_exec_all_in_price numeric,
    best_bid_price numeric,
    best_ask_price numeric,
    spread_ticks integer,

    visible_depth_to_qmax_dollars numeric,
    recent_qualifying_volume_3h_dollars numeric,
    route_rolling_cap_dollars numeric,
    route_cumulative_cap_remaining_dollars numeric,

    chosen boolean not null default false,
    route_decision_reason text,

    updated_at timestamptz not null default now(),

    constraint route_snapshots_outcome_check check (outcome_side in ('yes', 'no')),
    constraint route_snapshots_route_check check (route_name in ('BUY_YES_SELECTED', 'BUY_NO_OPPONENT')),
    unique (game_id, route_name, market_ticker, outcome_side)
);

-- =====================================================================
-- Order event log: every submit/fill/cancel/reject/skip.
-- =====================================================================
create table if not exists order_events (
    event_id uuid primary key default gen_random_uuid(),
    game_id text not null,
    market_ticker text,
    route_name text,
    order_id text,

    event_type text not null, -- submitted / filled / partial_fill / cancelled / cancel_requested / rejected / skipped
    order_mode text, -- passive / ioc / burst_ioc / cancel / shadow
    outcome_side text, -- yes / no

    price numeric,
    contracts integer,
    cost_dollars numeric,
    lead_hours numeric,

    reason text,
    raw_payload jsonb,
    created_at timestamptz not null default now(),

    constraint order_events_type_check check (event_type in ('submitted', 'filled', 'partial_fill', 'cancelled', 'cancel_requested', 'rejected', 'skipped')),
    constraint order_events_mode_check check (order_mode is null or order_mode in ('passive', 'ioc', 'burst_ioc', 'cancel', 'shadow')),
    constraint order_events_outcome_check check (outcome_side is null or outcome_side in ('yes', 'no'))
);

-- =====================================================================
-- Closed market summaries: one row per completed game.
-- =====================================================================
create table if not exists closed_market_summaries (
    game_id text primary key,
    game_date date,
    home_team text,
    away_team text,
    selected_team text,

    -- entered / did_not_enter / blocked / killed / expansion_gate / late_only / edge_failed / cancelled
    status text,
    did_enter boolean not null default false,

    model_prob_first_qualification numeric,
    market_prob_first_qualification numeric,
    max_abs_edge_observed numeric,
    max_norm_edge_observed numeric,
    final_q_max_price numeric,

    half_kelly_target_dollars numeric,
    portfolio_cap_dollars numeric,
    max_target_position_observed_dollars numeric,

    actual_entered_position_dollars numeric,
    average_fill_price numeric,
    vwap_lead_hours numeric,
    number_of_fills integer,
    number_of_order_attempts integer,

    outcome_win boolean,
    gross_payout_dollars numeric,
    pnl_dollars numeric,
    total_return numeric,
    log_return numeric,

    primary_binding_cap text,
    primary_reject_reason text,

    settled_at timestamptz
);

-- =====================================================================
-- Equity curve for historical view.
-- =====================================================================
create table if not exists equity_curve (
    ts timestamptz primary key,
    equity_dollars numeric not null,
    cash_dollars numeric,
    open_position_value_dollars numeric,
    realized_pnl_dollars numeric,
    drawdown_dollars numeric,
    total_markets_observed integer,
    entered_markets integer
);

-- =====================================================================
-- Bot heartbeat / health.
-- =====================================================================
create table if not exists bot_heartbeat (
    bot_id text primary key,
    status text not null,
    last_seen_at timestamptz not null,
    last_control_seen_at timestamptz,
    current_mode text,
    kalshi_connected boolean,
    market_data_connected boolean,
    database_connected boolean,
    open_orders_count integer,
    open_positions_count integer,
    last_error text
);

-- =====================================================================
-- Optional alert/event log.
-- =====================================================================
create table if not exists system_alerts (
    alert_id uuid primary key default gen_random_uuid(),
    severity text not null, -- info / warning / critical
    alert_type text not null,
    game_id text,
    message text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    acknowledged boolean not null default false,
    acknowledged_at timestamptz,
    acknowledged_by text,

    constraint system_alerts_severity_check check (severity in ('info', 'warning', 'critical'))
);

-- =====================================================================
-- Indexes.
-- =====================================================================
create index if not exists idx_live_market_snapshots_tipoff on live_market_snapshots (tipoff_ts);
create index if not exists idx_live_market_snapshots_phase on live_market_snapshots (phase);
create index if not exists idx_route_snapshots_game on route_snapshots (game_id);
create index if not exists idx_order_events_game_created on order_events (game_id, created_at desc);
create index if not exists idx_control_commands_received on control_commands (received_at desc);
create index if not exists idx_closed_market_summaries_date on closed_market_summaries (game_date desc);
create index if not exists idx_equity_curve_ts on equity_curve (ts);
create index if not exists idx_system_alerts_created on system_alerts (created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;
