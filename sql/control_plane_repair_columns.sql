alter table live_market_snapshots add column if not exists updated_at timestamptz not null default now();
alter table live_market_snapshots add column if not exists home_team text;
alter table live_market_snapshots add column if not exists away_team text;
alter table live_market_snapshots add column if not exists selected_team text;
alter table live_market_snapshots add column if not exists opponent_team text;
alter table live_market_snapshots add column if not exists tipoff_ts timestamptz;
alter table live_market_snapshots add column if not exists time_to_tipoff_minutes numeric;
alter table live_market_snapshots add column if not exists phase text;
alter table live_market_snapshots add column if not exists trading_status text;
alter table live_market_snapshots add column if not exists expansion_gate_status text;
alter table live_market_snapshots add column if not exists first_qualified_lead_hours numeric;
alter table live_market_snapshots add column if not exists model_prob numeric;
alter table live_market_snapshots add column if not exists market_prob numeric;
alter table live_market_snapshots add column if not exists abs_edge numeric;
alter table live_market_snapshots add column if not exists norm_edge numeric;
alter table live_market_snapshots add column if not exists q_max_price numeric;
alter table live_market_snapshots add column if not exists q_exec_all_in_price numeric;
alter table live_market_snapshots add column if not exists bankroll_for_sizing_dollars numeric;
alter table live_market_snapshots add column if not exists available_cash_after_buffer_dollars numeric;
alter table live_market_snapshots add column if not exists half_kelly_target_dollars numeric;
alter table live_market_snapshots add column if not exists portfolio_cap_dollars numeric;
alter table live_market_snapshots add column if not exists cash_cap_dollars numeric;
alter table live_market_snapshots add column if not exists target_position_now_dollars numeric;
alter table live_market_snapshots add column if not exists filled_position_dollars numeric;
alter table live_market_snapshots add column if not exists filled_contracts integer;
alter table live_market_snapshots add column if not exists reserved_open_order_dollars numeric;
alter table live_market_snapshots add column if not exists remaining_position_dollars numeric;
alter table live_market_snapshots add column if not exists visible_depth_cap_dollars numeric;
alter table live_market_snapshots add column if not exists recent_volume_cap_dollars numeric;
alter table live_market_snapshots add column if not exists cold_start_cap_dollars numeric;
alter table live_market_snapshots add column if not exists rolling_liquidity_cap_dollars numeric;
alter table live_market_snapshots add column if not exists cumulative_cap_remaining_dollars numeric;
alter table live_market_snapshots add column if not exists allowed_to_try_now_dollars numeric;
alter table live_market_snapshots add column if not exists next_child_order_dollars numeric;
alter table live_market_snapshots add column if not exists cash_limited_mode boolean not null default false;
alter table live_market_snapshots add column if not exists cash_priority_rule text;
alter table live_market_snapshots add column if not exists cash_priority_rank integer;
alter table live_market_snapshots add column if not exists cash_priority_rank_total integer;
alter table live_market_snapshots add column if not exists cash_priority_score numeric;
alter table live_market_snapshots add column if not exists expected_log_growth_next_child numeric;
alter table live_market_snapshots add column if not exists cash_priority_candidate_child_dollars numeric;
alter table live_market_snapshots add column if not exists cash_priority_allocated_child_dollars numeric;
alter table live_market_snapshots add column if not exists skipped_due_to_cash boolean not null default false;
alter table live_market_snapshots add column if not exists q_current_position numeric;
alter table live_market_snapshots add column if not exists q_avg_after_child numeric;
alter table live_market_snapshots add column if not exists target_position_binder text;
alter table live_market_snapshots add column if not exists execution_binder text;
alter table live_market_snapshots add column if not exists number_of_fills integer not null default 0;
alter table live_market_snapshots add column if not exists number_of_order_attempts integer not null default 0;
alter table live_market_snapshots add column if not exists number_of_trades_made integer not null default 0;
alter table live_market_snapshots add column if not exists average_fill_price numeric;
alter table live_market_snapshots add column if not exists vwap_lead_hours numeric;
alter table live_market_snapshots add column if not exists last_action text;
alter table live_market_snapshots add column if not exists last_reject_reason text;
alter table live_market_snapshots add column if not exists last_fill_ts timestamptz;
alter table live_market_snapshots add column if not exists last_order_ts timestamptz;
alter table live_market_snapshots add column if not exists market_data_ts timestamptz;
alter table live_market_snapshots add column if not exists model_snapshot_ts timestamptz;
alter table live_market_snapshots add column if not exists injury_data_ts timestamptz;
alter table live_market_snapshots add column if not exists orderbook_ts timestamptz;

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

alter table route_snapshots add column if not exists id uuid default gen_random_uuid();
alter table route_snapshots add column if not exists game_id text;
alter table route_snapshots add column if not exists route_name text;
alter table route_snapshots add column if not exists market_ticker text;
alter table route_snapshots add column if not exists outcome_side text;
alter table route_snapshots add column if not exists q_exec_all_in_price numeric;
alter table route_snapshots add column if not exists best_bid_price numeric;
alter table route_snapshots add column if not exists best_ask_price numeric;
alter table route_snapshots add column if not exists spread_ticks integer;
alter table route_snapshots add column if not exists visible_depth_to_qmax_dollars numeric;
alter table route_snapshots add column if not exists recent_qualifying_volume_3h_dollars numeric;
alter table route_snapshots add column if not exists route_rolling_cap_dollars numeric;
alter table route_snapshots add column if not exists route_cumulative_cap_remaining_dollars numeric;
alter table route_snapshots add column if not exists chosen boolean not null default false;
alter table route_snapshots add column if not exists route_decision_reason text;
alter table route_snapshots add column if not exists updated_at timestamptz not null default now();

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
    created_at timestamptz not null default now()
);

alter table order_events add column if not exists event_id uuid default gen_random_uuid();
alter table order_events add column if not exists game_id text;
alter table order_events add column if not exists market_ticker text;
alter table order_events add column if not exists route_name text;
alter table order_events add column if not exists order_id text;
alter table order_events add column if not exists event_type text;
alter table order_events add column if not exists order_mode text;
alter table order_events add column if not exists outcome_side text;
alter table order_events add column if not exists price numeric;
alter table order_events add column if not exists contracts integer;
alter table order_events add column if not exists cost_dollars numeric;
alter table order_events add column if not exists lead_hours numeric;
alter table order_events add column if not exists reason text;
alter table order_events add column if not exists raw_payload jsonb;
alter table order_events add column if not exists created_at timestamptz not null default now();

create table if not exists closed_market_summaries (
    game_id text primary key,
    game_date date,
    home_team text,
    away_team text,
    selected_team text,
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

create table if not exists system_alerts (
    alert_id uuid primary key default gen_random_uuid(),
    severity text not null,
    alert_type text not null,
    game_id text,
    message text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    acknowledged boolean not null default false,
    acknowledged_at timestamptz,
    acknowledged_by text
);

create index if not exists idx_live_market_snapshots_tipoff on live_market_snapshots (tipoff_ts);
create index if not exists idx_live_market_snapshots_phase on live_market_snapshots (phase);
create index if not exists idx_route_snapshots_game on route_snapshots (game_id);
create index if not exists idx_order_events_game_created on order_events (game_id, created_at desc);
create index if not exists idx_closed_market_summaries_date on closed_market_summaries (game_date desc);
create index if not exists idx_equity_curve_ts on equity_curve (ts);
create index if not exists idx_system_alerts_created on system_alerts (created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;

notify pgrst, 'reload schema';
