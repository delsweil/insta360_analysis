// app/api/admin/create-user/route.ts
//
// Creates a new Supabase Auth user and assigns them a role — an admin-only
// action. This has to be a server route rather than client code because it
// needs the Supabase *service role* key, which bypasses RLS entirely and
// must never be sent to the browser (unlike NEXT_PUBLIC_SUPABASE_ANON_KEY,
// this one has no NEXT_PUBLIC_ prefix, so Next.js won't bundle it client-side).
//
// Flow: admin submits the form on /admin/users → client sends the request
// here with their own session token → we verify that token actually belongs
// to an admin → only then do we use the service-role client to invite the
// new user and set their role.

import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY! // server-only — never NEXT_PUBLIC_

export async function POST(req: NextRequest) {
  try {
    const authHeader = req.headers.get('authorization')
    const token = authHeader?.replace('Bearer ', '')
    if (!token) {
      return NextResponse.json({ error: 'Missing auth token' }, { status: 401 })
    }

    // Verify the caller's own session using the plain anon-key client —
    // this just confirms who they are, using their own token/permissions.
    const callerClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: `Bearer ${token}` } },
    })
    const { data: { user: caller }, error: callerErr } = await callerClient.auth.getUser()
    if (callerErr || !caller) {
      return NextResponse.json({ error: 'Invalid session' }, { status: 401 })
    }

    const { data: callerRole } = await callerClient
      .from('user_roles')
      .select('role')
      .eq('user_id', caller.id)
      .single()

    if (callerRole?.role !== 'admin') {
      return NextResponse.json({ error: 'Admins only' }, { status: 403 })
    }

    const { email, displayName, role } = await req.json()
    if (!email || typeof email !== 'string') {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 })
    }
    if (!['admin', 'coach', 'player'].includes(role)) {
      return NextResponse.json({ error: 'Invalid role' }, { status: 400 })
    }

    // Only this client, built with the service role key, can create users
    // and bypass RLS. It's only ever constructed here, server-side.
    const adminClient = createClient(supabaseUrl, serviceRoleKey)

    // inviteUserByEmail sends the new user a link to set their own password —
    // no temporary password to generate or hand over insecurely.
    const { data: invited, error: inviteErr } = await adminClient.auth.admin.inviteUserByEmail(email)
    if (inviteErr || !invited?.user) {
      return NextResponse.json({ error: inviteErr?.message ?? 'Failed to invite user' }, { status: 500 })
    }

    const newUserId = invited.user.id

    const { error: roleErr } = await adminClient
      .from('user_roles')
      .upsert({ user_id: newUserId, role }, { onConflict: 'user_id' })
    if (roleErr) {
      return NextResponse.json({ error: `User invited, but role assignment failed: ${roleErr.message}` }, { status: 500 })
    }

    if (displayName) {
      await adminClient
        .from('profiles')
        .upsert({ id: newUserId, email, display_name: displayName }, { onConflict: 'id' })
    }

    return NextResponse.json({ success: true, userId: newUserId })
  } catch (err: any) {
    console.error('[create-user] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error' }, { status: 500 })
  }
}
