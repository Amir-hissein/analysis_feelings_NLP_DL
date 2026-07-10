// Session summary: counts of positive vs negative reviews analyzed so far.

function SessionHistogram({ historique }) {
  if (historique.length === 0) return null

  const positives = historique.filter((h) => h.label === 'positif').length
  const negatives = historique.length - positives
  const total = historique.length
  const avgConfidence = historique.reduce((s, h) => s + h.confiance, 0) / total
  const max = Math.max(positives, negatives, 1)

  const Bar = ({ label, count, color }) => (
    <div className="h-row">
      <div className="h-label">{label}</div>
      <div className="h-track">
        <div className="h-bar" style={{ width: `${(count / max) * 100}%`, background: color }} />
      </div>
      <div className="h-compte">{count}</div>
    </div>
  )

  return (
    <div className="histo">
      <h3>Your analyses (this session)</h3>
      <Bar label="Positive" count={positives} color="#16a34a" />
      <Bar label="Negative" count={negatives} color="#dc2626" />
      <div className="histo-stats">
        {total} reviews analyzed · average confidence {(avgConfidence * 100).toFixed(0)}%
      </div>
    </div>
  )
}

export default SessionHistogram
