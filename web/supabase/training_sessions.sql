-- Training sessions: a coach plans a session as a list of forms (exercises and small-sided games),
-- each with flexible text sections and one or more sketches (scenes).
-- Run once in the Supabase SQL editor.
--
-- Visibility: a session is visible only to its author until visibility = 'shared'.
-- Shared sessions can be read by every coach and admin of the club. Only the author edits.
-- (Players will see sessions through share links, added in a later step.)

create table if not exists training_sessions (
  id           uuid primary key default gen_random_uuid(),
  title        text not null default 'Neue Trainingseinheit',
  session_date date,
  focus        text not null default '',
  visibility   text not null default 'private' check (visibility in ('private', 'shared')),
  tags         text[] not null default '{}',   -- classification of every kind, age group included
  created_by   uuid not null default auth.uid() references auth.users (id) on delete cascade,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create table if not exists session_forms (
  id           uuid primary key default gen_random_uuid(),
  session_id   uuid not null references training_sessions (id) on delete cascade,
  position     integer not null default 0,
  kind         text not null default 'exercise' check (kind in ('exercise', 'spielform')),
  title        text not null default 'Neue Form',
  duration_min integer,
  players      text not null default '',
  -- ordered list of { id, key?, title, text }: flexible, the author adds, renames, reorders, deletes
  sections     jsonb not null default '[]'::jsonb,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create table if not exists form_scenes (
  id         uuid primary key default gen_random_uuid(),
  form_id    uuid not null references session_forms (id) on delete cascade,
  position   integer not null default 0,
  -- { type: 'pitch' | 'halfpitch' | 'grid' | 'blank', width_m?, height_m? }
  background jsonb not null default '{"type":"blank"}'::jsonb,
  -- same shape format as annotations.shapes, including the optional animation fields
  shapes     jsonb not null default '[]'::jsonb,
  caption    text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists session_forms_session_idx on session_forms (session_id, position);
create index if not exists form_scenes_form_idx on form_scenes (form_id, position);
create index if not exists training_sessions_author_idx on training_sessions (created_by);

-- Helpers. Security definer so that policies on the child tables do not trip over each other's row security.
create or replace function is_club_coach() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role in ('coach', 'admin'));
$$;

create or replace function can_read_session(sid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.training_sessions s
    where s.id = sid
      and (s.created_by = auth.uid() or (s.visibility = 'shared' and public.is_club_coach()))
  );
$$;

create or replace function owns_session(sid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.training_sessions s where s.id = sid and s.created_by = auth.uid());
$$;

create or replace function touch_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists training_sessions_touch on training_sessions;
create trigger training_sessions_touch before update on training_sessions for each row execute function touch_updated_at();
drop trigger if exists session_forms_touch on session_forms;
create trigger session_forms_touch before update on session_forms for each row execute function touch_updated_at();
drop trigger if exists form_scenes_touch on form_scenes;
create trigger form_scenes_touch before update on form_scenes for each row execute function touch_updated_at();

alter table training_sessions enable row level security;
alter table session_forms enable row level security;
alter table form_scenes enable row level security;

drop policy if exists "sessions: read" on training_sessions;
drop policy if exists "sessions: create" on training_sessions;
drop policy if exists "sessions: update own" on training_sessions;
drop policy if exists "sessions: delete own" on training_sessions;
create policy "sessions: read"       on training_sessions for select using (created_by = auth.uid() or (visibility = 'shared' and is_club_coach()));
create policy "sessions: create"     on training_sessions for insert with check (created_by = auth.uid() and is_club_coach());
create policy "sessions: update own" on training_sessions for update using (created_by = auth.uid()) with check (created_by = auth.uid());
create policy "sessions: delete own" on training_sessions for delete using (created_by = auth.uid());

drop policy if exists "forms: read" on session_forms;
drop policy if exists "forms: write own" on session_forms;
create policy "forms: read"      on session_forms for select using (can_read_session(session_id));
create policy "forms: write own" on session_forms for all using (owns_session(session_id)) with check (owns_session(session_id));

drop policy if exists "scenes: read" on form_scenes;
drop policy if exists "scenes: write own" on form_scenes;
create policy "scenes: read" on form_scenes for select using (
  exists (select 1 from session_forms f where f.id = form_id and can_read_session(f.session_id))
);
create policy "scenes: write own" on form_scenes for all
  using (exists (select 1 from session_forms f where f.id = form_id and owns_session(f.session_id)))
  with check (exists (select 1 from session_forms f where f.id = form_id and owns_session(f.session_id)));


-- Feldaufbau: how the forms are laid out on the pitch at the start of the session (one sketch per session).
alter table training_sessions add column if not exists layout jsonb;

-- Animation: how long to move into a sketch and how long to pause on it (seconds), per sketch.
alter table form_scenes add column if not exists timing jsonb;
