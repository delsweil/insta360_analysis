'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'
import Topbar from '@/components/Topbar'

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
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null) // userId pending confirmation
  const [deleting, setDeleting] = useState<string | null>(null)

  // Invite link state
  const [inviteRole, setInviteRole] = useState<'admin' | 'coach' | 'player'>('player')
  const [generatingLink, setGeneratingLink] = useState(false)
  const [inviteLink, setInviteLink] = useState('')
  const [inviteCopied, setInviteCopied] = useState(false)

  // Set-password state: which user's form is open, the typed password, status
  const [pwUser, setPwUser] = useState<string | null>(null)
  const [pwValue, setPwValue] = useState('')
  const [pwSaving, setPwSaving] = useState(false)
  const [pwDone, setPwDone] = useState<string | null>(null)
  const [pwError, setPwError] = useState('')

  const [newEmail, setNewEmail] = useState('')
  const [newDisplayName, setNewDisplayName] = useState('')
  const [newRole, setNewRole] = useState<'admin' | 'coach' | 'player'>('player')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState('')
  const [createSuccess, setCreateSuccess] = useState('')

  useEffect(() => {
    async function load() {
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

      const { data: roles } = await supabase
        .from('user_roles')
        .select('user_id, role')

      const { data: profiles } = await supabase
        .from('profiles')
        .select('id, email')

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
        const combined: UserWithRole[] = (roles ?? []).map(r => ({
          id: r.user_id,
          email: r.user_id,
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
      await supabase.from('user_roles').delete().eq('user_id', userId)
    } else {
      await supabase.from('user_roles').upsert(
        { user_id: userId, role: newRole },
        { onConflict: 'user_id' }
      )
    }
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u))
    setSaving(null)
  }

  const handleGenerateInvite = async () => {
    setGeneratingLink(true)
    setInviteLink('')
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setGeneratingLink(false); return }

    try {
      const res = await fetch('/api/admin/create-invite', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ role: inviteRole }),
      })
      const data = await res.json()
      if (res.ok && data.token) {
        setInviteLink(`${window.location.origin}/join/${data.token}`)
      } else {
        alert(`Could not create the link: ${data.error ?? 'unknown error'}`)
      }
    } catch {
      alert('Network error — please try again.')
    }
    setGeneratingLink(false)
  }

  const handleCopyInvite = async () => {
    await navigator.clipboard.writeText(inviteLink)
    setInviteCopied(true)
    setTimeout(() => setInviteCopied(false), 2000)
  }

  const openSetPassword = (userId: string) => {
    setPwUser(userId); setPwValue(''); setPwError('')
  }

  const generatePassword = () => {
    const chars = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    const bytes = crypto.getRandomValues(new Uint32Array(12))
    setPwValue(Array.from(bytes, b => chars[b % chars.length]).join(''))
  }

  const handleSetPassword = async (userId: string) => {
    if (pwValue.length < 8) { setPwError('At least 8 characters.'); return }
    setPwSaving(true); setPwError('')
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setPwSaving(false); return }
    try {
      const res = await fetch('/api/admin/set-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${session.access_token}` },
        body: JSON.stringify({ userId, password: pwValue }),
      })
      const result = await res.json()
      if (res.ok) {
        setPwDone(userId); setPwUser(null); setPwValue('')
        setTimeout(() => setPwDone(null), 4000)
      } else {
        setPwError(result.error ?? 'Failed.')
      }
    } catch {
      setPwError('Network error — please try again.')
    }
    setPwSaving(false)
  }

  const handleDeleteUser = async (userId: string) => {
    setDeleting(userId)
    setConfirmDelete(null)
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setDeleting(null); return }

    try {
      const res = await fetch('/api/admin/delete-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${session.access_token}` },
        body: JSON.stringify({ userId }),
      })
      const result = await res.json()
      if (res.ok) {
        setUsers(prev => prev.filter(u => u.id !== userId))
      } else {
        alert(`Delete failed: ${result.error}`)
      }
    } catch {
      alert('Network error — please try again.')
    }
    setDeleting(null)
  }

  const handleCreateUser = async () => {
    if (!newEmail.trim()) { setCreateError('Email is required'); return }
    setCreating(true)
    setCreateError('')
    setCreateSuccess('')

    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { setCreateError('Your session expired — please refresh and log in again.'); setCreating(false); return }

    try {
      const res = await fetch('/api/admin/create-user', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ email: newEmail.trim(), displayName: newDisplayName.trim(), role: newRole }),
      })
      const result = await res.json()

      if (!res.ok) {
        setCreateError(result.error || 'Failed to create user')
      } else {
        setCreateSuccess(`Invited ${newEmail} — they'll get an email to set their password.`)
        setUsers(prev => [...prev, {
          id: result.userId, email: newEmail.trim(), created_at: '', role: newRole,
        }])
        setNewEmail('')
        setNewDisplayName('')
        setNewRole('player')
      }
    } catch {
      setCreateError('Network error — please try again.')
    }
    setCreating(false)
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
      <Topbar role="admin" />

      <div style={{ padding: '24px 20px', maxWidth: 700, margin: '0 auto' }}>
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
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
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
              display: 'flex', flexDirection: 'column', gap: 10,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {/* Email + role label */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 14, fontWeight: 500, color: '#111318',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {user.email}
                  </div>
                  {user.role ? (
                    <div style={{
                      fontSize: 11, marginTop: 2,
                      color: ROLE_LABELS[user.role]?.color ?? '#8A8F9E',
                      fontWeight: 600,
                    }}>
                      {ROLE_LABELS[user.role]?.label}
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, marginTop: 2, color: '#8A8F9E' }}>No role assigned</div>
                  )}
                </div>

                {/* Role toggle buttons */}
                <div style={{ display: 'flex', gap: 5, flexShrink: 0 }}>
                  {(['admin', 'coach', 'player'] as const).map(role => (
                    <button
                      key={role}
                      onClick={() => handleRoleChange(user.id, user.role === role ? null : role)}
                      disabled={saving === user.id}
                      style={{
                        fontSize: 11, fontWeight: 600,
                        padding: '5px 12px', borderRadius: 99,
                        cursor: saving === user.id ? 'default' : 'pointer',
                        border: `1.5px solid ${user.role === role ? ROLE_LABELS[role].color : '#E4E6EE'}`,
                        background: user.role === role ? ROLE_LABELS[role].bg : '#fff',
                        color: user.role === role ? ROLE_LABELS[role].color : '#8A8F9E',
                        opacity: saving === user.id ? 0.6 : 1,
                      }}
                    >
                      {saving === user.id ? '…' : ROLE_LABELS[role].label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Action row: set password + delete */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingTop: 2, borderTop: '1px solid #F0F1F5' }}>
                {/* Set a new password (no email involved) */}
                {pwUser === user.id ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <input
                      type="text"
                      value={pwValue}
                      onChange={e => { setPwValue(e.target.value); setPwError('') }}
                      placeholder="New password (min 8)"
                      autoFocus
                      autoComplete="off"
                      style={{
                        fontSize: 12, padding: '4px 10px', borderRadius: 8, width: 170,
                        border: '1px solid #E4E6EE', outline: 'none', fontFamily: 'monospace',
                      }}
                    />
                    <button onClick={generatePassword} style={{
                      fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 99,
                      border: '1.5px solid #E4E6EE', background: '#fff', color: '#8A8F9E', cursor: 'pointer',
                    }}>Generate</button>
                    <button onClick={() => handleSetPassword(user.id)} disabled={pwSaving} style={{
                      fontSize: 11, fontWeight: 700, padding: '4px 12px', borderRadius: 99,
                      border: '1.5px solid #0f2972', background: '#0f2972', color: '#fff',
                      cursor: pwSaving ? 'default' : 'pointer', opacity: pwSaving ? 0.6 : 1,
                    }}>{pwSaving ? '…' : 'Save'}</button>
                    <button onClick={() => setPwUser(null)} style={{
                      fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 99,
                      border: '1.5px solid #E4E6EE', background: '#fff', color: '#8A8F9E', cursor: 'pointer',
                    }}>Cancel</button>
                    {pwError && <span style={{ fontSize: 11, color: '#b91c1c' }}>{pwError}</span>}
                  </div>
                ) : (
                  <button
                    onClick={() => openSetPassword(user.id)}
                    style={{
                      fontSize: 11, fontWeight: 600,
                      padding: '4px 12px', borderRadius: 99,
                      border: `1.5px solid ${pwDone === user.id ? '#86efac' : '#E4E6EE'}`,
                      background: pwDone === user.id ? '#f0fdf4' : '#fff',
                      color: pwDone === user.id ? '#166534' : '#8A8F9E',
                      cursor: 'pointer',
                    } as any}
                  >
                    {pwDone === user.id ? '✓ Password set' : '🔑 Set password'}
                  </button>
                )}

                {/* Delete — two-step confirm */}
                {confirmDelete === user.id ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 11, color: '#b91c1c', fontWeight: 600 }}>Delete permanently?</span>
                    <button
                      onClick={() => handleDeleteUser(user.id)}
                      disabled={deleting === user.id}
                      style={{
                        fontSize: 11, fontWeight: 700,
                        padding: '4px 12px', borderRadius: 99,
                        border: '1.5px solid #ef4444', background: '#fef2f2',
                        color: '#b91c1c', cursor: 'pointer',
                      }}
                    >
                      {deleting === user.id ? '…' : 'Yes, delete'}
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      style={{
                        fontSize: 11, fontWeight: 600,
                        padding: '4px 12px', borderRadius: 99,
                        border: '1.5px solid #E4E6EE', background: '#fff',
                        color: '#8A8F9E', cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmDelete(user.id)}
                    style={{
                      fontSize: 11, fontWeight: 600,
                      padding: '4px 12px', borderRadius: 99,
                      border: '1.5px solid #fca5a5', background: '#fff',
                      color: '#ef4444', cursor: 'pointer',
                    }}
                  >
                    🗑 Delete account
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Invite link */}
        <div style={{
          background: '#fff', border: '1px solid #E4E6EE',
          borderRadius: 12, padding: '16px 18px', marginTop: 16, marginBottom: 8,
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111318', marginBottom: 10 }}>
            Invite link
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <select
              value={inviteRole}
              onChange={e => { setInviteRole(e.target.value as typeof inviteRole); setInviteLink('') }}
              style={{
                fontSize: 13, padding: '8px 10px',
                border: '1px solid #E4E6EE', borderRadius: 8, outline: 'none',
                fontFamily: 'DM Sans, sans-serif', background: '#fff',
              }}
            >
              <option value="player">Player</option>
              <option value="coach">Coach</option>
              <option value="admin">Admin</option>
            </select>
            <button
              onClick={handleGenerateInvite}
              disabled={generatingLink}
              style={{
                fontSize: 13, fontWeight: 600, padding: '8px 18px',
                borderRadius: 8, border: 'none',
                background: generatingLink ? '#E4E6EE' : '#0f2972',
                color: generatingLink ? '#8A8F9E' : '#fff',
                cursor: generatingLink ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              {generatingLink ? 'Generating…' : 'Generate link'}
            </button>
          </div>

          {inviteLink && (
            <div style={{ marginTop: 10 }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                background: '#F8F8F6', border: '1px solid #E4E6EE',
                borderRadius: 8, padding: '8px 12px',
              }}>
                <span style={{
                  fontSize: 12, color: '#111318', flex: 1,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  fontFamily: 'monospace',
                }}>
                  {inviteLink}
                </span>
                <button
                  onClick={handleCopyInvite}
                  style={{
                    fontSize: 12, fontWeight: 600, padding: '4px 12px',
                    borderRadius: 6, border: '1.5px solid #0f2972',
                    background: inviteCopied ? '#0f2972' : '#fff',
                    color: inviteCopied ? '#fff' : '#0f2972',
                    cursor: 'pointer', flexShrink: 0,
                    fontFamily: 'DM Sans, sans-serif',
                  }}
                >
                  {inviteCopied ? '✓ Copied' : 'Copy'}
                </button>
              </div>
              <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 6 }}>
                Single-use · expires in 7 days · grants <strong>{inviteRole}</strong> access.
              </div>
            </div>
          )}
        </div>

        {/* Add user */}
        <div style={{
          background: '#fff', border: '1px solid #E4E6EE',
          borderRadius: 12, padding: '16px 18px', marginBottom: 16,
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#111318', marginBottom: 10 }}>
            Add user
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <input
              type="email" placeholder="Email" value={newEmail}
              onChange={e => setNewEmail(e.target.value)}
              style={{
                flex: '1 1 200px', fontSize: 13, padding: '8px 10px',
                border: '1px solid #E4E6EE', borderRadius: 8, outline: 'none',
                fontFamily: 'DM Sans, sans-serif',
              }}
            />
            <input
              type="text" placeholder="Display name (optional)" value={newDisplayName}
              onChange={e => setNewDisplayName(e.target.value)}
              style={{
                flex: '1 1 160px', fontSize: 13, padding: '8px 10px',
                border: '1px solid #E4E6EE', borderRadius: 8, outline: 'none',
                fontFamily: 'DM Sans, sans-serif',
              }}
            />
            <select
              value={newRole}
              onChange={e => setNewRole(e.target.value as 'admin' | 'coach' | 'player')}
              style={{
                fontSize: 13, padding: '8px 10px',
                border: '1px solid #E4E6EE', borderRadius: 8, outline: 'none',
                fontFamily: 'DM Sans, sans-serif', background: '#fff',
              }}
            >
              <option value="player">Player</option>
              <option value="coach">Coach</option>
              <option value="admin">Admin</option>
            </select>
            <button
              onClick={handleCreateUser}
              disabled={creating}
              style={{
                fontSize: 13, fontWeight: 600, padding: '8px 18px',
                borderRadius: 8, border: 'none',
                background: creating ? '#E4E6EE' : '#0f2972',
                color: creating ? '#8A8F9E' : '#fff',
                cursor: creating ? 'default' : 'pointer',
              }}
            >
              {creating ? 'Sending invite…' : 'Invite'}
            </button>
          </div>
          {createError && (
            <div style={{ fontSize: 12, color: '#b91c1c', marginTop: 8 }}>{createError}</div>
          )}
          {createSuccess && (
            <div style={{ fontSize: 12, color: '#166534', marginTop: 8 }}>{createSuccess}</div>
          )}
          <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 8 }}>
            They'll receive an email with a link to set their own password — no password to generate or share yourself.
          </div>
        </div>
      </div>
    </div>
  )
}
