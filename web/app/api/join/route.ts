// app/api/join/route.ts
//
// Validates an invite token and creates a new user account.
// Called from /join/[token] when a player submits the sign-up form.
//
// POST body: { token, email, password, displayName }
// - Atomically claims one use of the token (claim_invite) — safe when many
//   players open the same multi-use link at the same time
// - Creates Supabase auth user with the given email/password
// - Assigns the role attached to the token
// - Sets display_name in profiles
// - If anything fails after the claim, the account is rolled back and the use
//   is given back (release_invite), so a failed sign-up never burns the link
//
// Requires supabase/invite_multi_use.sql.

import { NextRequest, NextResponse } from 'next/server'
import { createClient, SupabaseClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const serviceKey  = process.env.SUPABASE_SERVICE_ROLE_KEY!

type InviteRow = {
  role: string
  expires_at: string
  used_at: string | null
  max_uses: number
  uses: number
}

// Explains why a token can't be used (only called after a failed claim, or by GET).
async function inviteProblem(
  adminClient: SupabaseClient,
  token: string,
): Promise<{ status: number; error: string } | null> {
  const { data: invite, error } = await adminClient
    .from('invite_tokens')
    .select('role, expires_at, used_at, max_uses, uses')
    .eq('token', token)
    .maybeSingle<InviteRow>()

  if (error || !invite) return { status: 404, error: 'Invalid or unknown invite link.' }
  if (new Date(invite.expires_at) < new Date()) return { status: 410, error: 'This invite link has expired.' }
  if (invite.uses >= invite.max_uses) {
    return {
      status: 409,
      error: invite.max_uses === 1
        ? 'This invite link has already been used.'
        : 'This invite link has reached its maximum number of sign-ups.',
    }
  }
  return null
}

export async function POST(req: NextRequest) {
  try {
    const { token, email, password, displayName } = await req.json()

    if (!token || !email || !password)
      return NextResponse.json({ error: 'token, email and password are required' }, { status: 400 })
    if (password.length < 8)
      return NextResponse.json({ error: 'Password must be at least 8 characters' }, { status: 400 })

    const adminClient = createClient(supabaseUrl, serviceKey)

    // 1. Claim one use atomically. Zero rows = unknown / expired / used up.
    const { data: claimed, error: claimErr } = await adminClient
      .rpc('claim_invite', { p_token: token })
    const claim = Array.isArray(claimed) ? claimed[0] : claimed
    if (claimErr || !claim) {
      const problem = await inviteProblem(adminClient, token)
      return NextResponse.json(
        { error: problem?.error ?? 'Invalid or unknown invite link.' },
        { status: problem?.status ?? 404 },
      )
    }
    const inviteId: string = claim.invite_id
    const role: string = claim.role

    // Undo the claim (and any partly created account) if a later step fails.
    const rollback = async (userId?: string) => {
      if (userId) await adminClient.auth.admin.deleteUser(userId)
      await adminClient.rpc('release_invite', { p_invite_id: inviteId })
    }

    // 2. Create auth user
    const { data: created, error: createErr } = await adminClient.auth.admin.createUser({
      email,
      password,
      email_confirm: true, // skip email confirmation — link itself is the gate
    })
    if (createErr || !created?.user) {
      await rollback()
      const taken = /already|registered|exists/i.test(createErr?.message ?? '')
      return NextResponse.json(
        { error: taken ? 'An account with this email already exists. Please sign in instead.'
                       : (createErr?.message ?? 'Failed to create account') },
        { status: taken ? 409 : 500 },
      )
    }
    const userId = created.user.id

    // 3. Assign role
    const { error: roleErr } = await adminClient.from('user_roles').insert({ user_id: userId, role })
    if (roleErr) {
      console.error('[join] role assignment failed:', roleErr)
      await rollback(userId)
      return NextResponse.json({ error: 'Could not finish creating your account. Please try again.' }, { status: 500 })
    }

    // 4. Set display name
    const { error: profileErr } = await adminClient.from('profiles').upsert(
      { id: userId, email, display_name: displayName?.trim() || null },
      { onConflict: 'id' },
    )
    if (profileErr) {
      console.error('[join] profile upsert failed:', profileErr)
      await adminClient.from('user_roles').delete().eq('user_id', userId)
      await rollback(userId)
      return NextResponse.json({ error: 'Could not finish creating your account. Please try again.' }, { status: 500 })
    }

    // 5. Audit trail (non-fatal)
    const { error: auditErr } = await adminClient
      .from('invite_uses').insert({ invite_id: inviteId, user_id: userId })
    if (auditErr) console.error('[join] could not record invite use:', auditErr)

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
  const problem = await inviteProblem(adminClient, token)
  if (problem) return NextResponse.json({ valid: false, error: problem.error })

  const { data: invite } = await adminClient
    .from('invite_tokens').select('role').eq('token', token).maybeSingle<{ role: string }>()
  return NextResponse.json({ valid: true, role: invite?.role })
}
