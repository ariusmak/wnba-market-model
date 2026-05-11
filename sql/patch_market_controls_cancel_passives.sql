alter table public.market_controls
    add column if not exists cancel_passive_orders boolean not null default false;

grant select, insert, update, delete on public.market_controls to service_role;

select pg_notify('pgrst', 'reload schema');
