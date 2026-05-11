create extension if not exists pgcrypto;

create table if not exists control_state (
    id text primary key default 'global',
    trading_enabled boolean not null default false,
    kill_switch_active boolean not null default false,
    allow_new_entries boolean not null default true,
    allow_ioc_orders boolean not null default true,
    allow_passive_orders boolean not null default true,
    allow_burst_mode boolean not null default true,
    mode text not null default 'normal',
    max_market_exposure_pct numeric not null default 0.15,
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

create table if not exists market_controls (
    game_id text primary key,
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

create table if not exists control_commands (
    command_id uuid primary key default gen_random_uuid(),
    command_type text not null,
    scope text not null,
    game_id text,
    payload jsonb not null default '{}'::jsonb,
    requested_by text not null,
    requested_via text not null,
    auth_status text not null,
    received_at timestamptz not null default now(),
    applied_at timestamptz,
    status text not null,
    reason text,
    constraint control_commands_scope_check check (scope in ('global', 'market')),
    constraint control_commands_status_check check (status in ('received', 'applied', 'rejected', 'failed'))
);

create index if not exists idx_control_commands_received on control_commands (received_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
grant usage, select on all sequences in schema public to service_role;
alter default privileges in schema public grant select, insert, update, delete on tables to service_role;
alter default privileges in schema public grant usage, select on sequences to service_role;
