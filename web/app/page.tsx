'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import { supabase, type Game, getCurrentUserRole, isCoachOrAdmin } from '@/lib/supabase'
import { useRouter } from 'next/navigation'
import Topbar from '@/components/Topbar'

export default function HomePage() {
  const [games, setGames] = useState<Game[]>([])
  const [user, setUser] = useState<any>(null)
  const [canDelete, setCanDelete] = useState(false)
  const [loading, setLoading] = useState(true)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        router.push('/login')
        return
      }
      setUser(user)

      const role = await getCurrentUserRole()
      setCanDelete(isCoachOrAdmin(role))

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

  const handleDeleteGame = async (gameId: string) => {
    await supabase.from('games').delete().eq('id', gameId)
    setGames(prev => prev.filter(g => g.id !== gameId))
    setConfirmDelete(null)
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

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar
        role={canDelete ? 'coach' : 'player'}
        email={user?.email}
        onSignOut={handleSignOut}
      />
      <div style={{ padding: '20px', maxWidth: 900, margin: '0 auto' }}>
      <div style={{ padding: '20px', maxWidth: 900, margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', marginBottom: 16,
        }}>
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 28, color: '#0f2972',
              letterSpacing: '0.02em', lineHeight: 1,
            }}>
              Spiele
            </div>
            <div style={{ fontSize: 12, color: '#8A8F9E', marginTop: 2 }}>
              {games.length} {games.length === 1 ? 'Spiel' : 'Spiele'} verfügbar
            </div>
          </div>
          {canDelete && (
            <button
              onClick={() => router.push('/add-game')}
              style={{
                fontSize: 12, fontWeight: 600,
                padding: '7px 16px', borderRadius: 8,
                border: 'none', background: '#0f2972',
                color: '#fff', cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              + Spiel hinzufügen
            </button>
          )}
        </div>

        {/* Games list */}
        {games.length === 0 ? (
          <div style={{
            background: '#fff', border: '1px solid #E4E6EE',
            borderRadius: 12, padding: '40px 20px',
            textAlign: 'center', color: '#8A8F9E', fontSize: 14,
          }}>
            Noch keine Spiele hochgeladen.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {games.map(game => (
              <div key={game.id}>
                <div
                  style={{
                    background: '#fff', border: '1px solid #E4E6EE',
                    borderRadius: confirmDelete === game.id ? '12px 12px 0 0' : 12,
                    padding: '14px 18px', cursor: 'pointer',
                    display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                  onMouseEnter={e => {
                    if (confirmDelete !== game.id)
                      e.currentTarget.style.boxShadow = '0 4px 20px rgba(15,41,114,0.1)'
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.boxShadow = 'none'
                  }}
                >
                  <div
                    style={{ flex: 1 }}
                    onClick={() => router.push(`/game/${game.id}`)}
                  >
                    <div style={{
                      fontFamily: 'Bebas Neue, sans-serif',
                      fontSize: 18, color: '#0f2972',
                      letterSpacing: '0.02em', lineHeight: 1.1,
                    }}>
                      {game.title}
                    </div>
                    <div style={{ fontSize: 12, color: '#8A8F9E', marginTop: 3 }}>
                      {new Date(game.date).toLocaleDateString('de-DE', {
                        day: '2-digit', month: 'long', year: 'numeric',
                      })}
                      {game.duration_sec && (
                        <> &nbsp;·&nbsp; {Math.floor(game.duration_sec / 60)} min</>
                      )}
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div
                      onClick={() => router.push(`/game/${game.id}`)}
                      style={{
                        fontSize: 11, fontWeight: 600,
                        color: '#0f2972', background: '#e8edf8',
                        padding: '4px 12px', borderRadius: 99,
                        cursor: 'pointer',
                      }}
                    >
                      Ansehen →
                    </div>
                    {canDelete && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setConfirmDelete(
                            confirmDelete === game.id ? null : game.id
                          )
                        }}
                        style={{
                          fontSize: 14, color: '#8A8F9E',
                          background: 'none', border: 'none',
                          cursor: 'pointer', padding: '4px 6px',
                          lineHeight: 1,
                        }}
                        title="Delete game"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>

                {/* Confirm delete panel */}
                {confirmDelete === game.id && (
                  <div style={{
                    background: '#fef2f2',
                    border: '1px solid #fca5a5',
                    borderTop: 'none',
                    borderRadius: '0 0 12px 12px',
                    padding: '12px 18px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 10,
                  }}>
                    <span style={{ fontSize: 13, color: '#991b1b' }}>
                      Spiel und alle Annotationen löschen?
                    </span>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        onClick={() => setConfirmDelete(null)}
                        style={{
                          fontSize: 12, fontWeight: 600,
                          padding: '5px 14px', borderRadius: 6,
                          border: '1px solid #fca5a5',
                          background: '#fff', color: '#991b1b',
                          cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                        }}
                      >
                        Abbrechen
                      </button>
                      <button
                        onClick={() => handleDeleteGame(game.id)}
                        style={{
                          fontSize: 12, fontWeight: 600,
                          padding: '5px 14px', borderRadius: 6,
                          border: 'none', background: '#ef4444',
                          color: '#fff', cursor: 'pointer',
                          fontFamily: 'DM Sans, sans-serif',
                        }}
                      >
                        Löschen
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
