'use client'

export const dynamic = 'force-dynamic'

import React from 'react'
import GameView from '@/components/GameView'

// A highlights link shows the same view as the game page, read-only and limited to the shared annotations.
export default function SharePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = React.use(params)
  return <GameView shareToken={token} />
}
