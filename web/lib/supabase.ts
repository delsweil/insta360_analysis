import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export type Game = {
  id: string
  title: string
  date: string
  video_url: string
  duration_sec: number
  uploaded_by: string
  created_at: string
}

export type Annotation = {
  id: string
  game_id: string
  user_id: string
  timestamp_sec: number
  end_timestamp_sec?: number
  label: string
  note?: string
  is_public: boolean
  created_at: string
}
