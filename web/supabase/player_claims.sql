-- "This is me": players mark themselves on the board; the tracker follows them and asks where it is unsure.
-- Run once in the Supabase SQL editor.
--
-- One row per player and game. `data` holds what the player said, as points (video time in board seconds, x, y in metres),
-- not as tracker ids, so the answers stay valid when the tracker changes. `tracker` is the version of the tracks file
-- the answers were given against.
-- Private: a player can read and change only their own row. Nobody else (coach included) can read it.
-- Rows with training_opt_in = true are meant for improving the tracker; read them with the service role,
-- and leave user_id out of what you export.
create table if not exists player_claims (
  id              uuid primary key default gen_random_uuid(),
  game_id         text not null,   -- games.id (text so it fits whatever type that column has)
  user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
  tracker         text,
  data            jsonb not null default '{}'::jsonb,
  training_opt_in boolean not null default false,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  unique (game_id, user_id)
);

alter table player_claims enable row level security;

drop policy if exists "own claims: read" on player_claims;
drop policy if exists "own claims: write" on player_claims;
create policy "own claims: read"   on player_claims for select using (user_id = auth.uid());
create policy "own claims: write"  on player_claims for all    using (user_id = auth.uid()) with check (user_id = auth.uid());
