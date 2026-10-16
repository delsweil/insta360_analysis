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

    const { userId } = await req.json()
    if (!userId || typeof userId !== 'string') {
      return NextResponse.json({ error: 'userId is required' }, { status: 400 })
    }
    if (userId === caller.id) {
      return NextResponse.json({ error: 'You cannot delete your own account' }, { status: 400 })
    }

    const adminClient = createClient(supabaseUrl, serviceRoleKey)

    const { error: roleErr } = await adminClient.from('user_roles').delete().eq('user_id', userId)
    if (roleErr) return NextResponse.json({ error: 'user_roles delete failed: ' + roleErr.message }, { status: 500 })

    const { error: profileErr } = await adminClient.from('profiles').delete().eq('id', userId)
    if (profileErr) return NextResponse.json({ error: 'profiles delete failed: ' + profileErr.message }, { status: 500 })

    const { error: deleteErr } = await adminClient.auth.admin.deleteUser(userId)
    if (deleteErr) return NextResponse.json({ error: 'auth delete failed: ' + deleteErr.message }, { status: 500 })

    return NextResponse.json({ success: true })
  } catch (err: any) {
    console.error('[delete-user] unexpected error:', err)
    return NextResponse.json({ error: 'Unexpected server error: ' + err.message }, { status: 500 })
  }
}
