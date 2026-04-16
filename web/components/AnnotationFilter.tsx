// components/AnnotationFilter.tsx
// Filter bar for annotation types with select all/none

'use client'

import AnnotationShape from './AnnotationShape'

export type FilterState = {
  goal: boolean
  shot: boolean
  setpiece: boolean
  tactical: boolean
}

export const ALL_FILTERS: FilterState = {
  goal: true, shot: true, setpiece: true, tactical: true,
}

export const NO_FILTERS: FilterState = {
  goal: false, shot: false, setpiece: false, tactical: false,
}

// Returns true if an annotation's label passes the filter
export function passesFilter(labelKey: string, filter: FilterState): boolean {
  if (labelKey.startsWith('goal')) return filter.goal
  if (labelKey.startsWith('shot')) return filter.shot
  if (labelKey.startsWith('setpiece')) return filter.setpiece
  if (labelKey === 'tactical') return filter.tactical
  return true
}

const FILTER_TYPES = [
  { key: 'goal',     label: 'Goal',      sampleKey: 'goal_home' },
  { key: 'shot',     label: 'Shot',      sampleKey: 'shot_home' },
  { key: 'setpiece', label: 'Set piece', sampleKey: 'setpiece_home' },
  { key: 'tactical', label: 'Tactical',  sampleKey: 'tactical' },
] as const

interface Props {
  filter: FilterState
  onChange: (filter: FilterState) => void
  counts?: Record<string, number>  // count per type
}

export default function AnnotationFilter({ filter, onChange, counts }: Props) {
  const allSelected = Object.values(filter).every(Boolean)
  const noneSelected = Object.values(filter).every(v => !v)

  const toggleAll = () => {
    onChange(allSelected ? NO_FILTERS : ALL_FILTERS)
  }

  const toggle = (key: keyof FilterState) => {
    onChange({ ...filter, [key]: !filter[key] })
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      flexWrap: 'wrap',
    }}>
      {/* Select all / none */}
      <button
        onClick={toggleAll}
        style={{
          fontSize: 10, fontWeight: 600,
          padding: '3px 10px', borderRadius: 99,
          border: '1.5px solid #E4E6EE',
          background: allSelected ? '#0f2972' : '#fff',
          color: allSelected ? '#fff' : '#8A8F9E',
          cursor: 'pointer', flexShrink: 0,
          fontFamily: 'DM Sans, sans-serif',
        }}
      >
        {allSelected ? 'All' : noneSelected ? 'None' : 'All'}
      </button>

      {FILTER_TYPES.map(({ key, label, sampleKey }) => {
        const active = filter[key]
        const count = counts?.[key] ?? 0
        return (
          <button
            key={key}
            onClick={() => toggle(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              fontSize: 10, fontWeight: 600,
              padding: '3px 10px', borderRadius: 99,
              border: `1.5px solid ${active ? '#E4E6EE' : '#F0F0F0'}`,
              background: active ? '#fff' : '#F8F8F6',
              color: active ? '#4A4F5C' : '#C0C4CE',
              cursor: 'pointer', flexShrink: 0,
              fontFamily: 'DM Sans, sans-serif',
              opacity: active ? 1 : 0.6,
              transition: 'all 0.15s',
            }}
          >
            <AnnotationShape
              labelKey={sampleKey}
              size={9}
              style={{ opacity: active ? 1 : 0.4 }}
            />
            {label}
            {count > 0 && (
              <span style={{
                fontSize: 9, color: '#8A8F9E',
                background: '#F8F8F6', borderRadius: 99,
                padding: '1px 5px', marginLeft: 1,
              }}>
                {count}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
