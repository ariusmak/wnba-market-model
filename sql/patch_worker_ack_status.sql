alter table public.bot_heartbeat
    add column if not exists last_control_seen_at timestamptz;

grant select, insert, update, delete on public.bot_heartbeat to service_role;

select pg_notify('pgrst', 'reload schema');
