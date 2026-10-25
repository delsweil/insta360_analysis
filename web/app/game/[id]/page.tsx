'use client'

export const dynamic = 'force-dynamic'

import React from 'react'
import GameView from '@/components/GameView'

export default function GamePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = React.use(params)
  return <GameView gameId={id} />
}
