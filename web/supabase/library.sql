-- Library of Übungen and Spielformen, organised by the game model. Run once in the Supabase SQL editor
-- (after training_sessions.sql).
--
-- A form stays in its author's session. Its library_status says where it stands:
--   private   → normal, only the author's session rules apply
--   proposed  → the author asked for it to be taken into the library; frozen, visible to admins for review
--   library   → approved by an admin; frozen, visible to every coach and admin of the club
--   archived  → taken out of the library by an admin; frozen, visible only to admins
-- Only admins set library / archived. An author changes a frozen form by withdrawing the proposal
-- (proposed → private) or, for a library form, by making a new version (a copy) in a session.

alter table session_forms add column if not exists concepts       text[] not null default '{}';  -- game-model ids, first = main concept
alter table session_forms add column if not exists library_status text   not null default 'private';
alter table session_forms add column if not exists library_note   text   not null default '';     -- the admin's comment when returning a proposal
alter table session_forms add column if not exists library_source uuid;                           -- the library form this one was copied from
alter table session_forms add column if not exists library_at     timestamptz;
alter table training_sessions add column if not exists concepts text[] not null default '{}';    -- focus of the session

alter table session_forms drop constraint if exists session_forms_library_status_check;
alter table session_forms add constraint session_forms_library_status_check check (library_status in ('private', 'proposed', 'library', 'archived'));
create index if not exists session_forms_library_idx on session_forms (library_status) where library_status <> 'private';

create or replace function is_admin() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role = 'admin');
$$;

create or replace function can_read_form(fid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.session_forms f
    where f.id = fid and (
      public.can_read_session(f.session_id)
      or (f.library_status = 'library' and public.is_club_coach())
      or (f.library_status in ('proposed', 'archived') and public.is_admin())
    )
  );
$$;

-- Authors edit their private forms; admins edit forms that are in the review/library process, never someone's private work.
create or replace function form_is_editable(fid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.session_forms f
    where f.id = fid and (
      (f.library_status = 'private' and public.owns_session(f.session_id))
      or (f.library_status <> 'private' and public.is_admin())
    )
  );
$$;

create or replace function session_has_library_form(sid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.session_forms f
    where f.session_id = sid and (
      (f.library_status = 'library' and public.is_club_coach())
      or (f.library_status in ('proposed', 'archived') and public.is_admin())
    )
  );
$$;

create or replace function session_has_frozen_form(sid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.session_forms f where f.session_id = sid and f.library_status in ('proposed', 'library'));
$$;

-- Who may change what.
create or replace function guard_library_status() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if public.is_admin() then
    if new.library_status is distinct from old.library_status or tg_op = 'INSERT' then new.library_at := now(); end if;
    return new;
  end if;
  if tg_op = 'INSERT' then
    new.library_status := 'private'; new.library_note := ''; new.library_at := null;
    return new;
  end if;
  -- update by someone who is not an admin
  if old.library_status <> 'private' then
    if old.library_status = 'proposed' and new.library_status = 'private' then
      new.library_note := old.library_note;
      return new;
    end if;
    raise exception 'Diese Form ist für die Bibliothek eingereicht oder aufgenommen und fixiert. Ziehe den Vorschlag zurück oder erstelle eine neue Version.';
  end if;
  if new.library_status not in ('private', 'proposed') then
    raise exception 'Nur Admins nehmen Formen in die Bibliothek auf.';
  end if;
  new.library_note := old.library_note;
  new.library_at := case when new.library_status = 'proposed' then now() else old.library_at end;
  return new;
end;
$$;
drop trigger if exists session_forms_guard on session_forms;
create trigger session_forms_guard before insert or update on session_forms for each row execute function guard_library_status();

-- Policies on forms
drop policy if exists "forms: read" on session_forms;
drop policy if exists "forms: write own" on session_forms;
drop policy if exists "forms: read library" on session_forms;
drop policy if exists "forms: write" on session_forms;
drop policy if exists "forms: insert own" on session_forms;
drop policy if exists "forms: update" on session_forms;
drop policy if exists "forms: delete" on session_forms;
create policy "forms: read" on session_forms for select using (
  can_read_session(session_id)
  or (library_status = 'library' and is_club_coach())
  or (library_status in ('proposed', 'archived') and is_admin())
);
create policy "forms: insert own" on session_forms for insert with check (owns_session(session_id));
create policy "forms: update" on session_forms for update
  using (owns_session(session_id) or (is_admin() and library_status <> 'private'))
  with check (owns_session(session_id) or is_admin());
create policy "forms: delete" on session_forms for delete
  using ((owns_session(session_id) and library_status = 'private') or (is_admin() and library_status <> 'private'));

-- Policies on scenes
drop policy if exists "scenes: read" on form_scenes;
drop policy if exists "scenes: write own" on form_scenes;
drop policy if exists "scenes: write" on form_scenes;
create policy "scenes: read" on form_scenes for select using (can_read_form(form_id));
create policy "scenes: write" on form_scenes for all
  using (form_is_editable(form_id)) with check (form_is_editable(form_id));

-- Sessions: a session that holds a library form can be opened (its other forms stay hidden by their own policy)
drop policy if exists "sessions: read" on training_sessions;
drop policy if exists "sessions: delete own" on training_sessions;
create policy "sessions: read" on training_sessions for select using (
  created_by = auth.uid()
  or (visibility = 'shared' and is_club_coach())
  or session_has_library_form(id)
);
create policy "sessions: delete own" on training_sessions for delete using (created_by = auth.uid() and not session_has_frozen_form(id));
