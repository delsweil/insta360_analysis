// Saves what a player marked ("This is me" points and answers) in the table player_claims (web/supabase/player_claims.sql).
import { useCallback, useEffect, useRef, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { EMPTY_CLAIM, type Claim, type LinkAnswer, type Pt } from '@/lib/tracks'

export type ClaimState = {
  signedIn: boolean | null          // null while checking
  claim: Claim
  optIn: boolean
  saving: boolean
  error: string | null
  addSeed: (p: Pt) => void
  addReject: (p: Pt) => void
  answer: (a: LinkAnswer) => void
  setOptIn: (v: boolean) => void
  reset: () => void
}

export function useClaim(gameId: string, tracker: string | null): ClaimState {
  const [signedIn, setSignedIn] = useState<boolean | null>(null)
  const [claim, setClaim] = useState<Claim>(EMPTY_CLAIM)
  const [optIn, setOptInState] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const userId = useRef<string | null>(null)
  const claimRef = useRef<Claim>(EMPTY_CLAIM)
  const optRef = useRef(false)

  useEffect(() => {
    let dead = false
    ;(async () => {
      const { data } = await supabase.auth.getUser()
      if (dead) return
      userId.current = data.user?.id ?? null
      setSignedIn(!!data.user)
      if (!data.user) return
      const { data: row, error: e } = await supabase.from('player_claims').select('data, training_opt_in').eq('game_id', gameId).eq('user_id', data.user.id).maybeSingle()
      if (dead) return
      if (e) { setError(/player_claims/.test(e.message) ? 'The table player_claims is missing: run web/supabase/player_claims.sql in the Supabase SQL editor.' : e.message); return }
      if (row) {
        const d = (row.data ?? {}) as Partial<Claim>
        const c = { seeds: d.seeds ?? [], rejects: d.rejects ?? [], links: d.links ?? [] }
        claimRef.current = c; optRef.current = !!row.training_opt_in
        setClaim(c)
        setOptInState(!!row.training_opt_in)
      }
    })()
    return () => { dead = true }
  }, [gameId])

  const save = useCallback(async (c: Claim, opt: boolean) => {
    if (!userId.current) return
    setSaving(true); setError(null)
    const { error: e } = await supabase.from('player_claims').upsert(
      { game_id: gameId, user_id: userId.current, tracker, data: c, training_opt_in: opt, updated_at: new Date().toISOString() },
      { onConflict: 'game_id,user_id' })
    setSaving(false)
    if (e) setError(/player_claims/.test(e.message) && /exist|relation|schema/i.test(e.message) ? 'The table player_claims is missing: run web/supabase/player_claims.sql in the Supabase SQL editor.' : e.message)
  }, [gameId, tracker])

  const update = useCallback((fn: (c: Claim) => Claim) => {
    const next = fn(claimRef.current)
    claimRef.current = next
    setClaim(next)
    void save(next, optRef.current)
  }, [save])

  return {
    signedIn, claim, optIn, saving, error,
    addSeed: p => update(c => ({ ...c, seeds: [...c.seeds, p] })),
    addReject: p => update(c => ({ ...c, rejects: [...c.rejects, p] })),
    answer: a => update(c => ({ ...c, links: [...c.links.filter(l => !(l.dir === a.dir && l.from.t === a.from.t && l.from.x === a.from.x && l.from.y === a.from.y)), a] })),
    setOptIn: v => { optRef.current = v; setOptInState(v); void save(claimRef.current, v) },
    reset: () => update(() => EMPTY_CLAIM),
  }
}
