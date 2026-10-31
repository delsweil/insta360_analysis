-- Multi-use invite links.
-- Run once in the Supabase SQL editor, AFTER the original invite_tokens migration
-- (the one in the comment at the top of app/api/admin/create-invite/route.ts).

alter table public.invite_tokens
  add column if not exists max_uses integer not null default 1
    check (max_uses between 1 and 500),
  add column if not exists uses integer not null default 0;

-- Tokens that were already consumed under the old single-use scheme count as one use.
update public.invite_tokens set uses = 1 where used_at is not null and uses = 0;

-- Audit trail: who signed up through which link.
create table if not exists public.invite_uses (
  id         uuid primary key default gen_random_uuid(),
  invite_id  uuid not null references public.invite_tokens(id) on delete cascade,
  user_id    uuid not null references auth.users(id) on delete cascade,
  used_at    timestamptz not null default now()
);
alter table public.invite_uses enable row level security;
-- No policies: only the service role (server routes) can read/write.

-- Atomically claim one use of a token. Returns zero rows if the token is
-- unknown, expired or used up. A single UPDATE means two people clicking at
-- the same moment can never push `uses` past `max_uses`.
create or replace function public.claim_invite(p_token uuid)
returns table (invite_id uuid, role text)
language sql
security definer
set search_path = public
as $$
  update public.invite_tokens t
     set uses    = t.uses + 1,
         used_at = case when t.uses + 1 >= t.max_uses then now() else t.used_at end
   where t.token = p_token
     and t.expires_at > now()
     and t.uses < t.max_uses
  returning t.id as invite_id, t.role as role;
$$;

-- Give a use back if account creation failed after the claim.
create or replace function public.release_invite(p_invite_id uuid)
returns void
language sql
security definer
set search_path = public
as $$
  update public.invite_tokens
     set uses    = greatest(uses - 1, 0),
         used_at = null
   where id = p_invite_id;
$$;

revoke all on function public.claim_invite(uuid)   from public, anon, authenticated;
revoke all on function public.release_invite(uuid) from public, anon, authenticated;
grant execute on function public.claim_invite(uuid)   to service_role;
grant execute on function public.release_invite(uuid) to service_role;
