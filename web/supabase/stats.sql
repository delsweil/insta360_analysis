-- Match statistics: one JSON file per game, "<game id>.json", in the storage bucket "stats".
-- Coaches and admins upload the analysis app's "*_chartdata.json" files on the add-game and edit-game screens;
-- everybody who can open the game or a share link can read it (team shape, territory and a heat map: no names, no faces).
-- Run once in the Supabase SQL editor.

insert into storage.buckets (id, name, public)
values ('stats', 'stats', true)
on conflict (id) do nothing;

create policy "stats are readable"
  on storage.objects for select
  using (bucket_id = 'stats');

create policy "coaches add stats"
  on storage.objects for insert to authenticated
  with check (
    bucket_id = 'stats'
    and exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role in ('coach', 'admin'))
  );

create policy "coaches replace stats"
  on storage.objects for update to authenticated
  using (
    bucket_id = 'stats'
    and exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role in ('coach', 'admin'))
  )
  with check (
    bucket_id = 'stats'
    and exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role in ('coach', 'admin'))
  );

create policy "coaches remove stats"
  on storage.objects for delete to authenticated
  using (
    bucket_id = 'stats'
    and exists (select 1 from public.user_roles r where r.user_id = auth.uid() and r.role in ('coach', 'admin'))
  );
