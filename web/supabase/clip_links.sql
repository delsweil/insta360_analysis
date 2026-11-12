-- Clip links: a share link that plays one stretch of the game and nothing else.
-- Run once in the Supabase SQL editor. Existing highlights links are unaffected (clip_only stays false).
alter table share_links
  add column if not exists clip_start_sec double precision,
  add column if not exists clip_end_sec   double precision,
  add column if not exists clip_only      boolean not null default false;
