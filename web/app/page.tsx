'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { supabase, type Game } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

export default function HomePage() {
  const [games, setGames] = useState<Game[]>([])
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        router.push('/login')
        return
      }
      setUser(user)

      const { data } = await supabase
        .from('games')
        .select('*')
        .order('date', { ascending: false })

      if (data) setGames(data)
      setLoading(false)
    }
    load()
  }, [router])

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.push('/login')
  }

  if (loading) return (
    <div style={{
      minHeight: '100vh',
      background: '#F8F8F6',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
      color: '#4A4F5C',
    }}>
      Loading...
    </div>
  )

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      {/* Topbar */}
      <div style={{
        background: '#0f2972',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 20,
          color: '#fff',
          letterSpacing: '0.05em',
        }}>
          Pfeil Phönix · Spielanalyse
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.7)' }}>
            {user?.email}
          </span>
          <button
            onClick={handleSignOut}
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
        </div>
      </div>

      <div style={{ padding: '20px 20px', maxWidth: 900, margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 16,
        }}>
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 28,
              color: '#0f2972',
              letterSpacing: '0.02em',
              lineHeight: 1,
            }}>
              Spiele
            </div>
            <div style={{ fontSize: 12, color: '#8A8F9E', marginTop: 2 }}>
              {games.length} {games.length === 1 ? 'Spiel' : 'Spiele'} verfügbar
            </div>
          </div>
        </div>

        {/* Games list */}
        {games.length === 0 ? (
          <div style={{
            background: '#fff',
            border: '1px solid #E4E6EE',
            borderRadius: 12,
            padding: '40px 20px',
            textAlign: 'center',
            color: '#8A8F9E',
            fontSize: 14,
          }}>
            Noch keine Spiele hochgeladen.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {games.map(game => (
              <div
                key={game.id}
                onClick={() => router.push(`/game/${game.id}`)}
                style={{
                  background: '#fff',
                  border: '1px solid #E4E6EE',
                  borderRadius: 12,
                  padding: '14px 18px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'box-shadow 0.15s',
                }}
                onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 4px 20px rgba(15,41,114,0.1)')}
                onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
              >
                <div>
                  <div style={{
                    fontFamily: 'Bebas Neue, sans-serif',
                    fontSize: 18,
                    color: '#0f2972',
                    letterSpacing: '0.02em',
                    lineHeight: 1.1,
                  }}>
                    {game.title}
                  </div>
                  <div style={{ fontSize: 12, color: '#8A8F9E', marginTop: 3 }}>
                    {new Date(game.date).toLocaleDateString('de-DE', {
                      day: '2-digit',
                      month: 'long',
                      year: 'numeric',
                    })}
                    {game.duration_sec && (
                      <>
                        &nbsp;·&nbsp;
                        {Math.floor(game.duration_sec / 60)} min
                      </>
                    )}
                  </div>
                </div>
                <div style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: '#0f2972',
                  background: '#e8edf8',
                  padding: '4px 12px',
                  borderRadius: 99,
                }}>
                  Ansehen →
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
