'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

type UserWithRole = {
  id: string
  email: string
  created_at: string
  role: 'admin' | 'coach' | 'player' | null
}

const ROLE_LABELS = {
  admin:  { label: 'Admin',  color: '#ef4444', bg: '#fef2f2' },
  coach:  { label: 'Coach',  color: '#E8780A', bg: '#FEF0E0' },
  player: { label: 'Player', color: '#0f2972', bg: '#e8edf8' },
}

export default function AdminUsersPage() {
  const router = useRouter()
  const [users, setUsers] = useState<UserWithRole[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState<string | null>(null)

  useEffect(() => {
    async function load() {
      // Verify current user is admin
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }

      const { data: roleData } = await supabase
        .from('user_roles')
        .select('role')
        .eq('user_id', user.id)
        .single()

      if (roleData?.role !== 'admin') {
        setError('Access denied — admins only.')
        setLoading(false)
        return
      }

      // Fetch all users + their roles via a join
      // We use the admin API via a server action ideally, but for simplicity
      // we fetch roles and combine with what we know
      const { data: roles } = await supabase
        .from('user_roles')
        .select('user_id, role')

      // Fetch user emails via auth.users — requires service role normally
      // Workaround: store emails in a public profiles table on signup
      // For now, fetch from profiles if it exists, otherwise show user_id
      const { data: profiles } = await supabase
        .from('profiles')
        .select('id, email')
        .catch(() => ({ data: null }))

      // Build combined list
      const roleMap: Record<string, string> = {}
      roles?.forEach(r => { roleMap[r.user_id] = r.role })

      if (profiles && profiles.length > 0) {
        const combined: UserWithRole[] = profiles.map((p: any) => ({
          id: p.id,
          email: p.email,
          created_at: '',
          role: (roleMap[p.id] ?? null) as any,
        }))
        setUsers(combined)
      } else {
        // Fallback: just show users from roles table with IDs
        const combined: UserWithRole[] = (roles ?? []).map(r => ({
          id: r.user_id,
          email: r.user_id, // will be replaced once profiles table exists
          created_at: '',
          role: r.role as any,
        }))
        setUsers(combined)
      }

      setLoading(false)
    }
    load()
  }, [router])

  const handleRoleChange = async (userId: string, newRole: 'admin' | 'coach' | 'player' | null) => {
    setSaving(userId)

    if (newRole === null) {
      // Remove role
      await supabase
        .from('user_roles')
        .delete()
        .eq('user_id', userId)
    } else {
      // Upsert role
      await supabase
        .from('user_roles')
        .upsert({ user_id: userId, role: newRole }, { onConflict: 'user_id' })
    }

    setUsers(prev => prev.map(u =>
      u.id === userId ? { ...u, role: newRole } : u
    ))
    setSaving(null)
  }

  if (loading) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C',
    }}>
      Loading...
    </div>
  )

  if (error) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      <div style={{
        background: '#fef2f2', border: '1px solid #fca5a5',
        borderRadius: 12, padding: '20px 28px',
        color: '#991b1b', fontSize: 14,
      }}>
        {error}
      </div>
    </div>
  )

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      {/* Topbar */}
      <div style={{
        background: '#0f2972',
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div
          style={{
            fontFamily: 'Bebas Neue, sans-serif',
            fontSize: 20, color: '#fff', letterSpacing: '0.05em',
            cursor: 'pointer',
          }}
          onClick={() => router.push('/')}
        >
          Pfeil Phönix · Spielanalyse
        </div>
        <div style={{
          background: '#ef4444', color: '#fff',
          fontSize: 10, fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          padding: '3px 10px', borderRadius: 99,
        }}>
          Admin
        </div>
      </div>

      <div style={{ padding: '24px 20px', maxWidth: 700, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif',
            fontSize: 28, color: '#0f2972',
            letterSpacing: '0.02em', lineHeight: 1,
          }}>
            Benutzerverwaltung
          </div>
          <div style={{ fontSize: 12, color: '#8A8F9E', marginTop: 4 }}>
            Rollen zuweisen · Admin kann alles · Coach kann annotieren · Player kann nur markieren
          </div>
        </div>

        {/* Role legend */}
        <div style={{
          display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap',
        }}>
          {Object.entries(ROLE_LABELS).map(([key, val]) => (
            <div key={key} style={{
              fontSize: 11, fontWeight: 600,
              padding: '4px 12px', borderRadius: 99,
              background: val.bg, color: val.color,
            }}>
              {val.label}
            </div>
          ))}
          <div style={{
            fontSize: 11, fontWeight: 600,
            padding: '4px 12px', borderRadius: 99,
            background: '#F8F8F6', color: '#8A8F9E',
            border: '1px solid #E4E6EE',
          }}>
            No role (read only)
          </div>
        </div>

        {/* Users list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {users.length === 0 && (
            <div style={{
              background: '#fff', border: '1px solid #E4E6EE',
              borderRadius: 12, padding: '30px 20px',
              textAlign: 'center', color: '#8A8F9E', fontSize: 14,
            }}>
              No users found. Invite users via Supabase → Authentication → Add user.
            </div>
          )}

          {users.map(user => (
            <div key={user.id} style={{
              background: '#fff', border: '1px solid #E4E6EE',
              borderRadius: 12, padding: '14px 18px',
              display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', gap: 12,
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 14, fontWeight: 500,
                  color: '#111318', overflow: 'hidden',
                  textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {user.email}
                </div>
                {user.role && (
                  <div style={{
                    fontSize: 11, marginTop: 2,
                    color: ROLE_LABELS[user.role]?.color ?? '#8A8F9E',
                    fontWeight: 600,
                  }}>
                    {ROLE_LABELS[user.role]?.label}
                  </div>
                )}
                {!user.role && (
                  <div style={{ fontSize: 11, marginTop: 2, color: '#8A8F9E' }}>
                    No role assigned
                  </div>
                )}
              </div>

              {/* Role selector */}
              <div style={{ display: 'flex', gap: 5, flexShrink: 0 }}>
                {(['admin', 'coach', 'player'] as const).map(role => (
                  <button
                    key={role}
                    onClick={() => handleRoleChange(
                      user.id,
                      user.role === role ? null : role
                    )}
                    disabled={saving === user.id}
                    style={{
                      fontSize: 11, fontWeight: 600,
                      padding: '5px 12px', borderRadius: 99,
                      cursor: saving === user.id ? 'default' : 'pointer',
                      border: `1.5px solid ${user.role === role
                        ? ROLE_LABELS[role].color
                        : '#E4E6EE'}`,
                      background: user.role === role
                        ? ROLE_LABELS[role].bg
                        : '#fff',
                      color: user.role === role
                        ? ROLE_LABELS[role].color
                        : '#8A8F9E',
                      transition: 'all 0.15s',
                      opacity: saving === user.id ? 0.6 : 1,
                    }}
                  >
                    {saving === user.id ? '...' : ROLE_LABELS[role].label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Note about inviting users */}
        <div style={{
          marginTop: 16, padding: '12px 16px',
          background: '#e8edf8', borderRadius: 10,
          fontSize: 12, color: '#0f2972', lineHeight: 1.6,
        }}>
          <strong>Neue Spieler einladen:</strong> Supabase → Authentication → Users → Add user.
          Nach dem ersten Login erscheinen sie hier und können eine Rolle erhalten.
        </div>
      </div>
    </div>
  )
}
