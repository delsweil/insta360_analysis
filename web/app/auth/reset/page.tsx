'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

export default function ResetPasswordPage() {
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [done, setDone] = useState(false)
  const router = useRouter()

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirm) { setError('Passwords do not match'); return }
    if (password.length < 8) { setError('Password must be at least 8 characters'); return }
    setSaving(true)
    const { error } = await supabase.auth.updateUser({ password })
    if (error) { setError(error.message); setSaving(false); return }
    setDone(true)
    setTimeout(() => router.push('/'), 2000)
  }

  return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      <div style={{
        background: '#fff', border: '1px solid #E4E6EE',
        borderRadius: 16, padding: '32px 28px', width: '100%', maxWidth: 400,
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 24, color: '#0f2972',
          letterSpacing: '0.04em', marginBottom: 20,
        }}>
          Reset Password
        </div>

        {done ? (
          <div style={{ color: '#22c55e', fontSize: 14 }}>
            Password updated — redirecting...
          </div>
        ) : (
          <form onSubmit={handleReset} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input
              type="password"
              placeholder="New password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              style={{
                padding: '10px 14px', borderRadius: 8,
                border: '1px solid #E4E6EE', fontSize: 14,
                fontFamily: 'DM Sans, sans-serif', outline: 'none',
              }}
            />
            <input
              type="password"
              placeholder="Confirm password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              required
              style={{
                padding: '10px 14px', borderRadius: 8,
                border: '1px solid #E4E6EE', fontSize: 14,
                fontFamily: 'DM Sans, sans-serif', outline: 'none',
              }}
            />
            {error && (
              <div style={{ fontSize: 12, color: '#ef4444' }}>{error}</div>
            )}
            <button
              type="submit"
              disabled={saving}
              style={{
                background: saving ? '#E4E6EE' : '#0f2972',
                color: saving ? '#8A8F9E' : '#fff',
                border: 'none', borderRadius: 8,
                padding: '11px', fontSize: 14, fontWeight: 600,
                cursor: saving ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              {saving ? 'Saving...' : 'Update password'}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}