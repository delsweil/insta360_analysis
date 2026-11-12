-- Clip links: a share link that plays one stretch of the game and nothing else.
-- Run once in the Supabase SQL editor. Existing highlights links are unaffected (clip_only stays false).
alter table share_links
  add column if not exists clip_start_sec double precision,
  add column if not exists clip_end_sec   double precision,
  add column if not exists clip_only      boolean not null default false;

-- Viewers of a shared link can cut a clip from it, without signing in.
-- The function checks the stretch against what the parent link shared, so nobody can make links to parts of the game that were not shared:
--   * parent is a clip link: the new clip must lie inside that clip;
--   * parent is a highlights link: inside a shared highlight, with 15 s of margin before and after (overlapping highlights count as one);
--   * at most 120 s long. Drawings of the parent's annotations that fall inside the clip travel with it. Expiry is inherited.
-- The margin and the length limit match CLIP_MARGIN_SEC and CLIP_MAX_SEC in web/lib/clipLinks.ts.
create or replace function create_clip_link(
  parent_token text, start_sec double precision, end_sec double precision, include_drawings boolean default true
) returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  p share_links%rowtype;
  v_ids share_links.annotation_ids%type;
  v_ok boolean;
  v_token text;
begin
  select * into p from share_links where token::text = parent_token;
  if not found then raise exception 'This share link does not exist.'; end if;
  if p.expires_at is not null and p.expires_at < now() then raise exception 'This share link has expired.'; end if;
  if start_sec is null or end_sec is null or end_sec - start_sec < 1 then raise exception 'The clip must be at least 1 second long.'; end if;
  if end_sec - start_sec > 120 then raise exception 'The clip can be 120 seconds at most.'; end if;

  if p.clip_only then
    v_ok := start_sec >= p.clip_start_sec and end_sec <= p.clip_end_sec;
  else
    with w as (
      select greatest(0, coalesce(a.context_start_sec, a.timestamp_sec) - 15) as a0,
             coalesce(a.end_timestamp_sec, a.timestamp_sec + 10) + 15 as b0
      from annotations a
      where a.game_id = p.game_id
        and (coalesce(cardinality(p.annotation_ids), 0) = 0
             or a.id::text = any (array(select x::text from unnest(p.annotation_ids) x)))
    ),
    m as (
      select a0, b0, max(b0) over (order by a0, b0 rows between unbounded preceding and 1 preceding) as prev_max from w
    ),
    g as (
      select a0, b0, sum(case when prev_max is null or a0 > prev_max then 1 else 0 end) over (order by a0, b0) as grp from m
    ),
    merged as (select min(a0) as a1, max(b0) as b1 from g group by grp)
    select exists (select 1 from merged where a1 <= start_sec and b1 >= end_sec) into v_ok;
  end if;
  if not v_ok then raise exception 'That part of the game was not shared.'; end if;

  v_ids := array(
    select a.id from annotations a
    where include_drawings
      and a.game_id = p.game_id
      and a.label = 'tactical'
      and a.shapes is not null and jsonb_array_length(a.shapes::jsonb) > 0
      and coalesce(a.annotation_end_sec, a.timestamp_sec + 1.5) >= start_sec
      and a.timestamp_sec <= end_sec
      and (coalesce(cardinality(p.annotation_ids), 0) = 0
           or a.id::text = any (array(select x::text from unnest(p.annotation_ids) x)))
  );

  insert into share_links (game_id, annotation_ids, expires_at, clip_start_sec, clip_end_sec, clip_only)
  values (p.game_id, v_ids, p.expires_at, round(start_sec::numeric, 1), round(end_sec::numeric, 1), true)
  returning token::text into v_token;
  return v_token;
end;
$$;

revoke all on function create_clip_link(text, double precision, double precision, boolean) from public;
grant execute on function create_clip_link(text, double precision, double precision, boolean) to anon, authenticated;
