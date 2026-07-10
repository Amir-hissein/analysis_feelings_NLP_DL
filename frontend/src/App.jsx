import { useState } from 'react'
import Contributions from './Contributions'
import SessionHistogram from './SessionHistogram'
import './App.css'

// Relative URL: FastAPI serves this app in production; Vite proxies it in dev.
const API_URL = '/predict'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  async function analyze() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error(`API error (${res.status})`)
      const data = await res.json()
      setResult(data)
      setHistory((h) => [...h, { label: data.label, confiance: data.confiance }])
    } catch {
      setError('Could not reach the API. Is it running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const isPositive = result?.label === 'positif'

  return (
    <div className="app">
      <header>
        <h1>🎬 Sentiment Analysis</h1>
        <p className="sous-titre">
          Paste a movie review — the model predicts whether it is positive or negative.
        </p>
      </header>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. This movie was absolutely brilliant, I loved every minute!"
        rows={6}
      />

      <button onClick={analyze} disabled={loading || text.trim().length === 0}>
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>

      {error && <div className="erreur">⚠️ {error}</div>}

      {result && (
        <div className={`resultat ${isPositive ? 'positif' : 'negatif'}`}>
          <div className="verdict">{isPositive ? '😊 Positive' : '😞 Negative'}</div>
          <div className="confiance-label">
            Confidence: {(result.confiance * 100).toFixed(1)}%
          </div>
          <div className="barre">
            <div className="barre-remplie" style={{ width: `${result.confiance * 100}%` }} />
          </div>

          <Contributions data={result.contributions} />
        </div>
      )}

      <SessionHistogram historique={history} />

      <footer>Model: TF-IDF + Logistic Regression with negation handling (89.3% accuracy)</footer>
    </div>
  )
}

export default App
