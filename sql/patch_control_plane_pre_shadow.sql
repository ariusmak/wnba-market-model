-- Consolidated pre-shadow control-plane patch.
--
-- Run this once in the Supabase SQL editor before the remote-control shadow
-- rehearsal. It is idempotent and combines the dashboard/worker columns needed
-- by:
--   - sql/patch_worker_ack_status.sql
--   - sql/patch_live_market_snapshots_columns.sql
--   - sql/patch_market_controls_cancel_passives.sql

alter table public.bot_heartbeat
    add column if not exists last_control_seen_at timestamptz;

alter table public.market_controls
    add column if not exists cancel_passive_orders boolean not null default false;

alter table public.live_market_snapshots add column if not exists opponent_team text;
alter table public.live_market_snapshots add column if not exists phase text;
alter table public.live_market_snapshots add column if not exists trading_status text;
alter table public.live_market_snapshots add column if not exists expansion_gate_status text;
alter table public.live_market_snapshots add column if not exists first_qualified_lead_hours numeric;
alter table public.live_market_snapshots add column if not exists model_prob numeric;
alter table public.live_market_snapshots add column if not exists model_prob_t20 numeric;
alter table public.live_market_snapshots add column if not exists model_prob_latest_pre_t8 numeric;
alter table public.live_market_snapshots add column if not exists model_prob_change_t20_to_t8 numeric;
alter table public.live_market_snapshots add column if not exists model_prob_changed_t20_to_t8 boolean not null default false;
alter table public.live_market_snapshots add column if not exists model_prob_last_refresh_at timestamptz;
alter table public.live_market_snapshots add column if not exists model_probability_update_count integer;
alter table public.live_market_snapshots add column if not exists market_prob numeric;
alter table public.live_market_snapshots add column if not exists abs_edge numeric;
alter table public.live_market_snapshots add column if not exists norm_edge numeric;
alter table public.live_market_snapshots add column if not exists q_max_price numeric;
alter table public.live_market_snapshots add column if not exists q_exec_all_in_price numeric;
alter table public.live_market_snapshots add column if not exists bankroll_for_sizing_dollars numeric;
alter table public.live_market_snapshots add column if not exists available_cash_after_buffer_dollars numeric;
alter table public.live_market_snapshots add column if not exists half_kelly_target_dollars numeric;
alter table public.live_market_snapshots add column if not exists portfolio_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists cash_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists target_position_now_dollars numeric;
alter table public.live_market_snapshots add column if not exists filled_position_dollars numeric;
alter table public.live_market_snapshots add column if not exists filled_contracts integer;
alter table public.live_market_snapshots add column if not exists reserved_open_order_dollars numeric;
alter table public.live_market_snapshots add column if not exists remaining_position_dollars numeric;
alter table public.live_market_snapshots add column if not exists visible_depth_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists recent_volume_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists cold_start_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists rolling_liquidity_cap_dollars numeric;
alter table public.live_market_snapshots add column if not exists cumulative_cap_remaining_dollars numeric;
alter table public.live_market_snapshots add column if not exists allowed_to_try_now_dollars numeric;
alter table public.live_market_snapshots add column if not exists next_child_order_dollars numeric;
alter table public.live_market_snapshots add column if not exists cash_limited_mode boolean not null default false;
alter table public.live_market_snapshots add column if not exists cash_priority_rule text;
alter table public.live_market_snapshots add column if not exists cash_priority_rank integer;
alter table public.live_market_snapshots add column if not exists cash_priority_rank_total integer;
alter table public.live_market_snapshots add column if not exists cash_priority_score numeric;
alter table public.live_market_snapshots add column if not exists expected_log_growth_next_child numeric;
alter table public.live_market_snapshots add column if not exists cash_priority_candidate_child_dollars numeric;
alter table public.live_market_snapshots add column if not exists cash_priority_allocated_child_dollars numeric;
alter table public.live_market_snapshots add column if not exists skipped_due_to_cash boolean not null default false;
alter table public.live_market_snapshots add column if not exists q_current_position numeric;
alter table public.live_market_snapshots add column if not exists q_avg_after_child numeric;
alter table public.live_market_snapshots add column if not exists target_position_binder text;
alter table public.live_market_snapshots add column if not exists execution_binder text;
alter table public.live_market_snapshots add column if not exists number_of_fills integer not null default 0;
alter table public.live_market_snapshots add column if not exists number_of_order_attempts integer not null default 0;
alter table public.live_market_snapshots add column if not exists number_of_trades_made integer not null default 0;
alter table public.live_market_snapshots add column if not exists average_fill_price numeric;
alter table public.live_market_snapshots add column if not exists vwap_lead_hours numeric;
alter table public.live_market_snapshots add column if not exists last_action text;
alter table public.live_market_snapshots add column if not exists last_reject_reason text;
alter table public.live_market_snapshots add column if not exists last_fill_ts timestamptz;
alter table public.live_market_snapshots add column if not exists last_order_ts timestamptz;
alter table public.live_market_snapshots add column if not exists market_data_ts timestamptz;
alter table public.live_market_snapshots add column if not exists model_snapshot_ts timestamptz;
alter table public.live_market_snapshots add column if not exists injury_data_ts timestamptz;
alter table public.live_market_snapshots add column if not exists orderbook_ts timestamptz;

create index if not exists idx_live_market_snapshots_tipoff on public.live_market_snapshots (tipoff_ts);
create index if not exists idx_live_market_snapshots_phase on public.live_market_snapshots (phase);

grant select, insert, update, delete on public.bot_heartbeat to service_role;
grant select, insert, update, delete on public.market_controls to service_role;
grant select, insert, update, delete on public.live_market_snapshots to service_role;

select pg_notify('pgrst', 'reload schema');
