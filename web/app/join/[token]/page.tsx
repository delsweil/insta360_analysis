'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import React from 'react'

interface Props {
  params: Promise<{ token: string }>
}

export default function JoinPage({ params }: Props) {
  const { token } = React.use(params)
  const router = useRouter()

  const [validating, setValidating] = useState(true)
  const [tokenError, setTokenError] = useState('')
  const [role, setRole] = useState('')

  const [displayName, setDisplayName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState('')
  const [done, setDone] = useState(false)

  // Validate the token on mount
  useEffect(() => {
    fetch(`/api/join?token=${encodeURIComponent(token)}`)
      .then(r => r.json())
      .then(data => {
        if (data.valid) {
          setRole(data.role)
        } else {
          setTokenError(data.error ?? 'Invalid invite link.')
        }
        setValidating(false)
      })
      .catch(() => {
        setTokenError('Could not validate invite link. Please try again.')
        setValidating(false)
      })
  }, [token])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError('')

    if (!displayName.trim()) { setFormError('Please enter your name.'); return }
    if (password.length < 8) { setFormError('Password must be at least 8 characters.'); return }
    if (password !== confirm) { setFormError('Passwords do not match.'); return }

    setSubmitting(true)
    try {
      const res = await fetch('/api/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, email, password, displayName: displayName.trim() }),
      })
      const data = await res.json()
      if (!res.ok) {
        setFormError(data.error ?? 'Something went wrong.')
        setSubmitting(false)
        return
      }
      setDone(true)
      setTimeout(() => router.push('/login'), 3000)
    } catch {
      setFormError('Network error — please try again.')
      setSubmitting(false)
    }
  }

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '10px 14px',
    border: '1px solid #E4E6EE', borderRadius: 8,
    fontSize: 14, fontFamily: 'DM Sans, sans-serif',
    outline: 'none', color: '#111318', boxSizing: 'border-box',
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f2972',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
      padding: 16,
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
          fontSize: 28, color: '#0f2972',
          letterSpacing: '0.04em', marginBottom: 4,
        }}>
          ASN Pfeil Phönix
        </div>
        <div style={{ fontSize: 13, color: '#8A8F9E', marginBottom: 28 }}>
          Spielanalyse · Konto erstellen
        </div>

        {validating && (
          <div style={{ color: '#8A8F9E', fontSize: 14, textAlign: 'center', padding: '20px 0' }}>
            Einladung wird geprüft…
          </div>
        )}

        {!validating && tokenError && (
          <div style={{
            background: '#fef2f2', border: '1px solid #fecaca',
            borderRadius: 10, padding: '16px 18px',
            color: '#991b1b', fontSize: 13, lineHeight: 1.5,
          }}>
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Ungültiger Einladungslink</div>
            {tokenError}
            <div style={{ marginTop: 12, fontSize: 12, color: '#b91c1c' }}>
              Bitte wende dich an den Administrator für einen neuen Link.
            </div>
          </div>
        )}

        {!validating && !tokenError && !done && (
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {role && (
              <div style={{
                fontSize: 11, fontWeight: 700,
                padding: '4px 12px', borderRadius: 99,
                background: '#e8edf8', color: '#0f2972',
                display: 'inline-block', marginBottom: 4,
                alignSelf: 'flex-start',
              }}>
                Eingeladen als: {role === 'player' ? 'Spieler' : role === 'coach' ? 'Trainer' : 'Admin'}
              </div>
            )}
            <input
              type="text"
              placeholder="Dein Name (wird angezeigt)"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              required
              autoFocus
              style={inputStyle}
            />
            <input
              type="email"
              placeholder="E-Mail-Adresse"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              style={inputStyle}
            />
            <input
              type="password"
              placeholder="Passwort (min. 8 Zeichen)"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              style={inputStyle}
            />
            <input
              type="password"
              placeholder="Passwort bestätigen"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              required
              style={inputStyle}
            />

            {formError && (
              <div style={{ fontSize: 12, color: '#ef4444' }}>{formError}</div>
            )}

            <button
              type="submit"
              disabled={submitting}
              style={{
                padding: '10px',
                background: submitting ? '#E4E6EE' : '#0f2972',
                color: submitting ? '#8A8F9E' : '#fff',
                border: 'none', borderRadius: 8,
                fontSize: 14, fontWeight: 600,
                cursor: submitting ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
                marginTop: 4,
              }}
            >
              {submitting ? 'Konto wird erstellt…' : 'Konto erstellen'}
            </button>

            <div style={{ fontSize: 11, color: '#8A8F9E', textAlign: 'center', marginTop: 4 }}>
              Du hast bereits ein Konto?{' '}
              <a href="/login" style={{ color: '#0f2972', fontWeight: 600 }}>Anmelden</a>
            </div>
          </form>
        )}

        {done && (
          <div style={{
            background: '#f0fdf4', border: '1px solid #bbf7d0',
            borderRadius: 10, padding: '16px 18px',
            color: '#166534', fontSize: 13, textAlign: 'center', lineHeight: 1.6,
          }}>
            <div style={{ fontSize: 20, marginBottom: 8 }}>✓</div>
            <div style={{ fontWeight: 700 }}>Konto erstellt!</div>
            Du wirst zur Anmeldeseite weitergeleitet…
          </div>
        )}
      </div>
    </div>
  )
}
