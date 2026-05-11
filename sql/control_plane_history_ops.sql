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
    acknowledged_by text,
    constraint system_alerts_severity_check check (severity in ('info', 'warning', 'critical'))
);

create index if not exists idx_closed_market_summaries_date on closed_market_summaries (game_date desc);
create index if not exists idx_equity_curve_ts on equity_curve (ts);
create index if not exists idx_system_alerts_created on system_alerts (created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;
