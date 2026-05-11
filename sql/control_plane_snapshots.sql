create table if not exists live_market_snapshots (
    game_id text primary key,
    updated_at timestamptz not null default now(),
    home_team text,
    away_team text,
    selected_team text,
    opponent_team text,
    tipoff_ts timestamptz,
    time_to_tipoff_minutes numeric,
    phase text,
    trading_status text,
    expansion_gate_status text,
    first_qualified_lead_hours numeric,
    model_prob numeric,
    market_prob numeric,
    abs_edge numeric,
    norm_edge numeric,
    q_max_price numeric,
    q_exec_all_in_price numeric,
    bankroll_for_sizing_dollars numeric,
    available_cash_after_buffer_dollars numeric,
    half_kelly_target_dollars numeric,
    portfolio_cap_dollars numeric,
    cash_cap_dollars numeric,
    target_position_now_dollars numeric,
    filled_position_dollars numeric,
    filled_contracts integer,
    reserved_open_order_dollars numeric,
    remaining_position_dollars numeric,
    visible_depth_cap_dollars numeric,
    recent_volume_cap_dollars numeric,
    cold_start_cap_dollars numeric,
    rolling_liquidity_cap_dollars numeric,
    cumulative_cap_remaining_dollars numeric,
    allowed_to_try_now_dollars numeric,
    next_child_order_dollars numeric,
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
    target_position_binder text,
    execution_binder text,
    number_of_fills integer not null default 0,
    number_of_order_attempts integer not null default 0,
    number_of_trades_made integer not null default 0,
    average_fill_price numeric,
    vwap_lead_hours numeric,
    last_action text,
    last_reject_reason text,
    last_fill_ts timestamptz,
    last_order_ts timestamptz,
    market_data_ts timestamptz,
    model_snapshot_ts timestamptz,
    injury_data_ts timestamptz,
    orderbook_ts timestamptz,
    constraint live_market_price_checks check (
        (model_prob is null or (model_prob >= 0 and model_prob <= 1)) and
        (market_prob is null or (market_prob >= 0 and market_prob <= 1)) and
        (q_max_price is null or (q_max_price >= 0 and q_max_price <= 1)) and
        (q_exec_all_in_price is null or (q_exec_all_in_price >= 0 and q_exec_all_in_price <= 1)) and
        (q_current_position is null or (q_current_position >= 0 and q_current_position <= 1)) and
        (q_avg_after_child is null or (q_avg_after_child >= 0 and q_avg_after_child <= 1))
    )
);

create table if not exists route_snapshots (
    id uuid primary key default gen_random_uuid(),
    game_id text not null,
    route_name text not null,
    market_ticker text not null,
    outcome_side text not null,
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

create table if not exists order_events (
    event_id uuid primary key default gen_random_uuid(),
    game_id text not null,
    market_ticker text,
    route_name text,
    order_id text,
    event_type text not null,
    order_mode text,
    outcome_side text,
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

create index if not exists idx_live_market_snapshots_tipoff on live_market_snapshots (tipoff_ts);
create index if not exists idx_live_market_snapshots_phase on live_market_snapshots (phase);
create index if not exists idx_route_snapshots_game on route_snapshots (game_id);
create index if not exists idx_order_events_game_created on order_events (game_id, created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;
