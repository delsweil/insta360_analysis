import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL as string
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY as string
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY as string

export async function POST(req: NextRequest) {
  try {
    const authHeader = req.headers.get('authorization')
    const token = authHeader?.replace('Bearer ', '')
    if (!token) return NextResponse.json({ error: 'Missing auth token' }, { status: 401 })
    const callerClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: `Bearer ${token}` } },
    })
    const { data: { user: caller }, error: callerErr } = await callerClient.auth.getUser()
    if (callerErr || !caller) return NextResponse.json({ error: 'Invalid session' }, { status: 401 })
    const { data: callerRole } = await callerClient
      .from('user_roles').select('role').eq('user_id', caller.id).single()
    if (callerRole?.role !== 'admin') return NextResponse.json({ error: 'Admins only' }, { status: 403 })
    const { email } = await req.json()
    if (!email || typeof email !== 'string') {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 })
    }
    const adminClient = createClient(supabaseUrl, serviceRoleKey)
    const appUrl = process.env.NEXT_PUBLIC_APP_URL || supabaseUrl
    const { error: resetErr } = await adminClient.auth.resetPasswordForEmail(email, {
      redirectTo: appUrl + '/auth/callback?next=/profile',
    })
    if (resetErr) return NextResponse.json({ error: resetErr.message }, { status: 500 })
    return NextResponse.json({ success: true })
  } catch (err: any) {
    console.error('[reset-password] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error' }, { status: 500 })
  }
}
