'use client'

import { useRouter } from 'next/navigation'
import Image from 'next/image'

interface TopbarProps {
  role?: 'admin' | 'coach' | 'player' | null
  email?: string
  onSignOut?: () => void
  showBack?: boolean
  backHref?: string
}

export default function Topbar({
  role,
  email,
  onSignOut,
  showBack,
  backHref = '/',
}: TopbarProps) {
  const router = useRouter()

  const rolePill = role === 'admin'
    ? { label: 'Admin', bg: '#ef4444' }
    : role === 'coach'
    ? { label: 'Coach', bg: '#E8780A' }
    : null

  return (
    <div style={{
      background: '#0f2972',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 20px',
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          cursor: showBack || backHref ? 'pointer' : 'default',
        }}
        onClick={() => router.push(backHref)}
      >
        <Image
          src="/asn-logo.svg"
          alt="ASN Pfeil Phönix"
          width={32}
          height={32}
          style={{ filter: 'brightness(0) invert(1)', flexShrink: 0 }}
        />
        <div style={{ lineHeight: 1.15 }}>
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif',
            fontSize: 18,
            color: '#fff',
            letterSpacing: '0.05em',
            lineHeight: 1,
          }}>
            ASN Pfeil Phönix
          </div>
          <div style={{
            fontSize: 9,
            color: 'rgba(255,255,255,0.4)',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
          }}>
            Spielanalyse · Nürnberg
          </div>
        </div>
      </div>

      {/* Right side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {email && (
          <span style={{
            fontSize: 12,
            color: 'rgba(255,255,255,0.85)',
            fontWeight: 500,
          }}>
            {email}
          </span>
        )}
        {rolePill && (
          <div style={{
            background: rolePill.bg,
            color: '#fff',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            padding: '3px 10px',
            borderRadius: 99,
          }}>
            {rolePill.label}
          </div>
        )}
        {role === 'admin' || role === 'coach' ? (
          <button
            onClick={() => router.push('/admin/users')}
            style={{
              fontSize: 11,
              padding: '4px 12px',
              borderRadius: 99,
              border: '1px solid rgba(255,255,255,0.3)',
              background: 'transparent',
              color: '#fff',
              cursor: 'pointer',
              fontFamily: 'DM Sans, sans-serif',
            }}
          >
            Admin
          </button>
        ) : null}
        {onSignOut && (
          <button
            onClick={onSignOut}
            style={{
              fontSize: 11,
              padding: '4px 12px',
              borderRadius: 99,
              border: '1px solid rgba(255,255,255,0.3)',
              background: 'transparent',
              color: '#fff',
              cursor: 'pointer',
              fontFamily: 'DM Sans, sans-serif',
            }}
          >
            Sign out
          </button>
        )}
      </div>
    </div>
  )
}
