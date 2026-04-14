'use client'

export const dynamic = 'force-dynamic'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) {
      setError(error.message)
      setLoading(false)
    } else {
      router.push('/')
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f2972',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      <div style={{
        background: '#fff',
        borderRadius: 16,
        padding: '40px 36px',
        width: '100%',
        maxWidth: 400,
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 28,
          color: '#0f2972',
          letterSpacing: '0.04em',
          marginBottom: 4,
        }}>
          Pfeil Phönix
        </div>
        <div style={{ fontSize: 13, color: '#8A8F9E', marginBottom: 28 }}>
          Spielanalyse · Anmelden
        </div>
        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <input
            type="email"
            placeholder="E-Mail"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            style={{
              padding: '10px 14px',
              border: '1px solid #E4E6EE',
              borderRadius: 8,
              fontSize: 14,
              fontFamily: 'DM Sans, sans-serif',
              outline: 'none', color: '#111318',
            }}
          />
          <input
            type="password"
            placeholder="Passwort"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            style={{
              padding: '10px 14px',
              border: '1px solid #E4E6EE',
              borderRadius: 8,
              fontSize: 14,
              fontFamily: 'DM Sans, sans-serif',
              outline: 'none', color: '#111318',
            }}
          />
          {error && (
            <div style={{ fontSize: 12, color: '#ef4444' }}>{error}</div>
          )}
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: '10px',
              background: loading ? '#E4E6EE' : '#0f2972',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: loading ? 'default' : 'pointer',
              fontFamily: 'DM Sans, sans-serif',
              marginTop: 4,
            }}
          >
            {loading ? 'Anmelden...' : 'Anmelden'}
          </button>
        </form>
      </div>
    </div>
  )
}
