import { useState } from 'react'
import type { RunState } from '../api'
import { kindMeta } from '../designer/eventKinds'
import { PauseIcon, PlayIcon, StepIcon } from './icons'

export const SPEEDS = [1, 2, 5, 10, 25, 50]

interface Props {
  state: RunState
  playing: boolean
  speed: number
  busy: boolean
  decisionDays: number[]
  onPlay: (playing: boolean) => void
  onSpeed: (speed: number) => void
  onStep: () => void
  onRunTo: (day: number) => void
}

export default function Timeline({ state, playing, speed, busy, decisionDays, onPlay, onSpeed, onStep, onRunTo }: Props) {
  const { start_day, end_day } = state.scenario.timeline
  const span = end_day - start_day + 1
  const done = Math.min(1, (state.day - start_day) / span)
  const at = (d: number) => `${Math.max(0, Math.min(100, ((d - start_day) / span) * 100))}%`
  const [target, setTarget] = useState('')
  const targetDay = Number(target)

  return (
    <div className="timeline" onClick={(e) => e.stopPropagation()}>
      <button className="btn icon yellow" aria-label={playing ? 'Pause' : 'Play'} disabled={state.finished}
        onClick={() => onPlay(!playing)}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      <button className="btn icon" aria-label="Advance one day" disabled={state.finished || playing || busy} onClick={onStep}>
        <StepIcon />
      </button>
      <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
        <span>Speed</span>
        <select className="select" value={speed} onChange={(e) => onSpeed(Number(e.target.value))} aria-label="Days per second">
          {SPEEDS.map((s) => <option key={s} value={s}>{s} day{s > 1 ? 's' : ''}/s</option>)}
        </select>
      </label>
      <span className="day">{state.finished ? `Finished, day ${end_day}` : `Day ${state.day}`}</span>
      <div className="track" aria-hidden="true">
        {state.events.map((e) => (
          <div key={e.id} className="event-band" title={e.name}
            style={{ left: at(e.start_day), width: `calc(${at((e.end_day ?? end_day) + 1)} - ${at(e.start_day)})`, background: kindMeta(e.kind).color }} />
        ))}
        <div className="fill" style={{ width: `${done * 100}%`, opacity: 0.85 }} />
        {decisionDays.map((d) => <div key={d} className="tick" style={{ left: at(d), opacity: 0.35 }} />)}
      </div>
      <span className="day muted">{end_day}</span>
      <form style={{ display: 'flex', gap: 6 }} onSubmit={(e) => {
        e.preventDefault()
        if (targetDay > state.day - 1) { onRunTo(targetDay); setTarget('') }
      }}>
        <input className="input" style={{ width: 92 }} inputMode="numeric" placeholder="to day" aria-label="Run to the end of day"
          value={target} onChange={(e) => setTarget(e.target.value)} disabled={state.finished || playing} />
        <button className="btn" type="submit" disabled={state.finished || playing || busy || !(targetDay >= state.day)}>Run to</button>
      </form>
    </div>
  )
}
