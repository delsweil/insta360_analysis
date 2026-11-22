// "This is me", state shared by the video overlay and the board panel: the board, its tracks, what the player marked.
import { useCallback, useMemo, useState } from 'react'
import { type Board, type BoardState } from '@/lib/board'
import { useClaim, type ClaimState } from '@/lib/claims'
import { useFollowed, useTracks, pointAt, type Followed, type Question, type Tracks, type TracksState } from '@/lib/tracks'

export type Me = {
  board: Board | null
  tracks: Tracks | null
  tracksState: TracksState
  reloadTracks: () => void
  claim: ClaimState
  followed: Followed | null
  picking: boolean
  setPicking: (v: boolean) => void
  question: Question | null          // the next open question that was not skipped
  openQuestions: number
  skip: (key: string) => void
  resetSkips: () => void
  /** set while the player is tapping themselves; undefined otherwise */
  onPick?: (moment: number, k: number) => void
}

export function useSelf(gameId: string, state: BoardState): Me {
  const board = state.status === 'ready' ? state.board : null
  const { state: tracksState, reload: reloadTracks } = useTracks(gameId, board)
  const tracks = tracksState.status === 'ready' ? tracksState.tracks : null
  const claim = useClaim(gameId, tracks?.tracker ?? null)
  const followed = useFollowed(board, tracks, claim.claim)
  const [picking, setPicking] = useState(false)
  const [skipped, setSkipped] = useState<Set<string>>(new Set())
  const skip = useCallback((key: string) => setSkipped(s => new Set(s).add(key)), [])
  const resetSkips = useCallback(() => setSkipped(new Set()), [])
  const pickFn = useMemo(() => {
    if (!picking || !board || !tracks || claim.signedIn !== true) return undefined
    return (m: number, k: number) => {
      const pid = tracks.fp[m][k]
      const pt = pointAt(board, m, k)
      if (followed?.status.has(pid)) claim.addReject(pt); else claim.addSeed(pt)
      setPicking(false)
    }
  }, [picking, board, tracks, claim, followed])
  const open = followed ? followed.questions.filter(q => !skipped.has(q.key)) : []
  return { board, tracks, tracksState, reloadTracks, claim, followed, picking, setPicking, question: open[0] ?? null, openQuestions: open.length, skip, resetSkips, onPick: pickFn }
}
