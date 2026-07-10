import { useState } from 'react'
import Contributions from './Contributions'
import SessionHistogram from './SessionHistogram'
import './App.css'

// Adresse de notre API FastAPI (Phase 11). En prod, on la mettrait dans une
// variable d'environnement ; ici on la code en dur pour rester simple.
const API_URL = 'http://127.0.0.1:8000/predict'

function App() {
  // ── L'ÉTAT du composant (sa "mémoire") ────────────────────────────
  // Chaque appel à setXxx() redessine automatiquement l'interface.
  const [texte, setTexte] = useState('')        // le texte tapé par l'utilisateur
  const [resultat, setResultat] = useState(null) // la réponse de l'API {label, confiance...}
  const [chargement, setChargement] = useState(false) // true pendant l'appel API
  const [erreur, setErreur] = useState(null)     // message d'erreur éventuel
  const [historique, setHistorique] = useState([]) // tous les résultats de la session

  // ── L'appel à l'API (asynchrone) ──────────────────────────────────
  async function analyser() {
    setChargement(true)
    setErreur(null)
    setResultat(null)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: texte }),
      })
      if (!res.ok) throw new Error(`Erreur API (${res.status})`)
      const data = await res.json()   // { label, proba_positif, confiance, contributions }
      setResultat(data)
      // On ajoute ce résultat à l'historique (pour l'histogramme de session).
      setHistorique((h) => [...h, { label: data.label, confiance: data.confiance }])
    } catch (e) {
      // Souvent : l'API n'est pas lancée. On l'indique clairement.
      setErreur("Impossible de contacter l'API. Est-elle bien lancée sur le port 8000 ?")
    } finally {
      setChargement(false)
    }
  }

  const estPositif = resultat?.label === 'positif'

  // ── Le rendu (JSX = du HTML dans du JS) ────────────────────────────
  return (
    <div className="app">
      <header>
        <h1>🎬 Analyse de sentiments</h1>
        <p className="sous-titre">
          Colle un avis de film (en anglais) — le modèle devine s'il est positif ou négatif.
        </p>
      </header>

      <textarea
        value={texte}
        onChange={(e) => setTexte(e.target.value)}  // à chaque frappe -> met à jour l'état
        placeholder="Ex : This movie was absolutely brilliant, I loved every minute!"
        rows={6}
      />

      <button
        onClick={analyser}
        disabled={chargement || texte.trim().length === 0}  // désactivé si vide ou en cours
      >
        {chargement ? 'Analyse en cours…' : 'Analyser'}
      </button>

      {/* Affichage conditionnel : on ne montre le bloc que s'il y a quelque chose */}
      {erreur && <div className="erreur">⚠️ {erreur}</div>}

      {resultat && (
        <div className={`resultat ${estPositif ? 'positif' : 'negatif'}`}>
          <div className="verdict">
            {estPositif ? '😊 Positif' : '😞 Négatif'}
          </div>
          <div className="confiance-label">
            Confiance : {(resultat.confiance * 100).toFixed(1)} %
          </div>
          {/* Barre de confiance : sa largeur = le pourcentage */}
          <div className="barre">
            <div
              className="barre-remplie"
              style={{ width: `${resultat.confiance * 100}%` }}
            />
          </div>

          {/* Le graphique des contributions (le "pourquoi") */}
          <Contributions data={resultat.contributions} />
        </div>
      )}

      {/* Histogramme récapitulatif de tous les avis testés cette session */}
      <SessionHistogram historique={historique} />

      <footer>Modèle : TF-IDF + Régression Logistique (88,6 % de précision)</footer>
    </div>
  )
}

export default App
