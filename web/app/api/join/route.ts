// app/api/join/route.ts
//
// Validates an invite token and creates a new user account.
// Called from /join/[token] when a player submits the sign-up form.
//
// POST body: { token, email, password, displayName }
// - Validates token exists, not expired, not already used
// - Creates Supabase auth user with the given email/password
// - Assigns the role attached to the token
// - Sets display_name in profiles
// - Marks the token as used

import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const serviceKey  = process.env.SUPABASE_SERVICE_ROLE_KEY!

export async function POST(req: NextRequest) {
  try {
    const { token, email, password, displayName } = await req.json()

    if (!token || !email || !password)
      return NextResponse.json({ error: 'token, email and password are required' }, { status: 400 })
    if (password.length < 8)
      return NextResponse.json({ error: 'Password must be at least 8 characters' }, { status: 400 })

    const adminClient = createClient(supabaseUrl, serviceKey)

    // Validate token
    const { data: invite, error: fetchErr } = await adminClient
      .from('invite_tokens')
      .select('id, role, expires_at, used_at')
      .eq('token', token)
      .single()

    if (fetchErr || !invite)
      return NextResponse.json({ error: 'Invalid or unknown invite link.' }, { status: 404 })
    if (invite.used_at)
      return NextResponse.json({ error: 'This invite link has already been used.' }, { status: 409 })
    if (new Date(invite.expires_at) < new Date())
      return NextResponse.json({ error: 'This invite link has expired.' }, { status: 410 })

    // Create auth user
    const { data: created, error: createErr } = await adminClient.auth.admin.createUser({
      email,
      password,
      email_confirm: true, // skip email confirmation — link itself is the gate
    })
    if (createErr || !created?.user)
      return NextResponse.json({ error: createErr?.message ?? 'Failed to create account' }, { status: 500 })

    const userId = created.user.id

    // Assign role
    await adminClient.from('user_roles').insert({ user_id: userId, role: invite.role })

    // Set display name
    await adminClient.from('profiles').upsert(
      { id: userId, email, display_name: displayName?.trim() || null },
      { onConflict: 'id' }
    )

    // Mark token used
    await adminClient
      .from('invite_tokens')
      .update({ used_at: new Date().toISOString(), used_by: userId })
      .eq('id', invite.id)

    return NextResponse.json({ success: true })
  } catch (err: any) {
    console.error('[join] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error' }, { status: 500 })
  }
}

// GET: validate token without consuming it — used by the join page on load
// to show a useful error before the player fills in the form.
export async function GET(req: NextRequest) {
  const token = req.nextUrl.searchParams.get('token')
  if (!token) return NextResponse.json({ error: 'Missing token' }, { status: 400 })

  const adminClient = createClient(supabaseUrl, serviceKey)
  const { data: invite, error } = await adminClient
    .from('invite_tokens')
    .select('role, expires_at, used_at')
    .eq('token', token)
    .single()

  if (error || !invite) return NextResponse.json({ valid: false, error: 'Invalid invite link.' })
  if (invite.used_at)    return NextResponse.json({ valid: false, error: 'This invite link has already been used.' })
  if (new Date(invite.expires_at) < new Date()) return NextResponse.json({ valid: false, error: 'This invite link has expired.' })

  return NextResponse.json({ valid: true, role: invite.role })
}
