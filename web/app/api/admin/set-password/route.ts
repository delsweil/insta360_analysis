// app/api/admin/set-password/route.ts
//
// An admin sets a new password for a user directly (no reset email).
// POST { userId, password }  — Authorization: Bearer <admin's session token>
// Uses the service-role key on the server only.

import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL as string
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY as string
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY as string

export async function POST(req: NextRequest) {
  try {
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

    const { userId, password } = await req.json()
    if (!userId || typeof userId !== 'string')
      return NextResponse.json({ error: 'userId is required' }, { status: 400 })
    if (typeof password !== 'string' || password.length < 8)
      return NextResponse.json({ error: 'Password must be at least 8 characters' }, { status: 400 })

    const adminClient = createClient(supabaseUrl, serviceRoleKey)
    const { error } = await adminClient.auth.admin.updateUserById(userId, { password })
    if (error) return NextResponse.json({ error: error.message }, { status: 500 })

    return NextResponse.json({ success: true })
  } catch (err: any) {
    console.error('[set-password] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error' }, { status: 500 })
  }
}
