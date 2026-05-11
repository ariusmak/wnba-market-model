create table if not exists public.route_snapshots (
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

create table if not exists public.order_events (
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

create table if not exists public.closed_market_summaries (
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

create table if not exists public.equity_curve (
    ts timestamptz primary key,
    equity_dollars numeric not null,
    cash_dollars numeric,
    open_position_value_dollars numeric,
    realized_pnl_dollars numeric,
    drawdown_dollars numeric,
    total_markets_observed integer,
    entered_markets integer
);

create table if not exists public.bot_heartbeat (
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

create table if not exists public.system_alerts (
    alert_id uuid primary key default gen_random_uuid(),
    severity text not null,
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

create index if not exists idx_route_snapshots_game on public.route_snapshots (game_id);
create index if not exists idx_order_events_game_created on public.order_events (game_id, created_at desc);
create index if not exists idx_closed_market_summaries_date on public.closed_market_summaries (game_date desc);
create index if not exists idx_equity_curve_ts on public.equity_curve (ts);
create index if not exists idx_system_alerts_created on public.system_alerts (created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;

select pg_notify('pgrst', 'reload schema');
