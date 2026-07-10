// Diverging bar chart showing which words drove the prediction.
// Right/green = pushes positive, left/red = pushes negative; bar length = |weight|.

function Contributions({ data }) {
  if (!data || data.length === 0) return null

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.contribution)))

  return (
    <div className="contribs">
      <h3>Why this decision?</h3>
      <p className="contribs-aide">
        Most decisive words (simplified form). ▶ green = positive · ◀ red = negative.
      </p>

      {data.map((d) => {
        const positive = d.contribution > 0
        const width = `${(Math.abs(d.contribution) / maxAbs) * 100}%`
        return (
          <div className="c-row" key={d.mot}>
            <div className="c-mot">{d.mot}</div>
            <div className="c-track">
              <div className="c-half c-left">
                {!positive && <div className="c-bar c-neg" style={{ width }} />}
              </div>
              <div className="c-half c-right">
                {positive && <div className="c-bar c-pos" style={{ width }} />}
              </div>
            </div>
            <div className={`c-val ${positive ? 'pos' : 'neg'}`}>
              {positive ? '+' : ''}{d.contribution.toFixed(2)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default Contributions
