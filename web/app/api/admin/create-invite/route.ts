// app/api/admin/create-invite/route.ts
//
// Generates a single-use invite token that admins can share as a URL.
// The token is stored in the invite_tokens table with an expiry and a role.
// When a player visits /join/[token] and signs up, the token is consumed.
//
// Requires: the invite_tokens table (see SQL migration below).
//
// SQL to run in Supabase SQL editor:
//
//   create table public.invite_tokens (
//     id          uuid primary key default gen_random_uuid(),
//     token       uuid not null default gen_random_uuid(),
//     role        text not null default 'player' check (role in ('admin','coach','player')),
//     created_by  uuid not null references auth.users(id),
//     expires_at  timestamptz not null default (now() + interval '7 days'),
//     used_at     timestamptz,
//     used_by     uuid references auth.users(id)
//   );
//
//   alter table public.invite_tokens enable row level security;
//   -- Only service role (server routes) can read/write — no client access.

import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const anonKey    = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

export async function POST(req: NextRequest) {
  try {
    // Verify caller is admin
    const token = req.headers.get('authorization')?.replace('Bearer ', '')
    if (!token) return NextResponse.json({ error: 'Missing auth token' }, { status: 401 })

    const callerClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: `Bearer ${token}` } },
    })
    const { data: { user: caller }, error: callerErr } = await callerClient.auth.getUser()
    if (callerErr || !caller) return NextResponse.json({ error: 'Invalid session' }, { status: 401 })

    const { data: callerRole } = await callerClient
      .from('user_roles').select('role').eq('user_id', caller.id).single()
    if (callerRole?.role !== 'admin') return NextResponse.json({ error: 'Admins only' }, { status: 403 })

    const { role = 'player', expiryDays = 7 } = await req.json()
    if (!['admin', 'coach', 'player'].includes(role))
      return NextResponse.json({ error: 'Invalid role' }, { status: 400 })

    const adminClient = createClient(supabaseUrl, serviceKey)

    const expiresAt = new Date()
    expiresAt.setDate(expiresAt.getDate() + expiryDays)

    const { data, error } = await adminClient
      .from('invite_tokens')
      .insert({ role, created_by: caller.id, expires_at: expiresAt.toISOString() })
      .select('token')
      .single()

    if (error || !data) return NextResponse.json({ error: error?.message ?? 'Failed to create token' }, { status: 500 })

    return NextResponse.json({ token: data.token })
  } catch (err: any) {
    console.error('[create-invite] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error' }, { status: 500 })
  }
}
