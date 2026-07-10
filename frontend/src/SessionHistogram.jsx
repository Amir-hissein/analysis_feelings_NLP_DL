// Histogramme de session : récapitule tous les avis analysés depuis l'ouverture.
// Reçoit `historique` = liste de { label, confiance }.

function SessionHistogram({ historique }) {
  if (historique.length === 0) return null

  // Calculs à partir de l'historique.
  const positifs = historique.filter((h) => h.label === 'positif').length
  const negatifs = historique.length - positifs
  const total = historique.length
  const confMoy = historique.reduce((s, h) => s + h.confiance, 0) / total
  const maxi = Math.max(positifs, negatifs, 1) // évite la division par 0

  // Une barre horizontale (largeur = compte / max).
  const Barre = ({ libelle, compte, couleur }) => (
    <div className="h-row">
      <div className="h-label">{libelle}</div>
      <div className="h-track">
        <div
          className="h-bar"
          style={{ width: `${(compte / maxi) * 100}%`, background: couleur }}
        />
      </div>
      <div className="h-compte">{compte}</div>
    </div>
  )

  return (
    <div className="histo">
      <h3>Tes analyses (cette session)</h3>
      <Barre libelle="Positifs" compte={positifs} couleur="#16a34a" />
      <Barre libelle="Négatifs" compte={negatifs} couleur="#dc2626" />
      <div className="histo-stats">
        {total} avis analysés · confiance moyenne {(confMoy * 100).toFixed(0)} %
      </div>
    </div>
  )
}

export default SessionHistogram
