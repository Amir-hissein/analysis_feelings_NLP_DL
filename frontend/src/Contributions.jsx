// Graphique des contributions : montre POURQUOI le modèle a décidé.
// Barres divergentes : à droite (vert) = pousse vers positif, à gauche (rouge) = négatif.
// La longueur d'une barre = importance du mot (|contribution|).

function Contributions({ data }) {
  if (!data || data.length === 0) return null

  // Échelle : la plus grande contribution (en valeur absolue) = barre pleine.
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.contribution)))

  return (
    <div className="contribs">
      <h3>Pourquoi cette décision ?</h3>
      <p className="contribs-aide">
        Mots les plus décisifs (forme simplifiée). ▶ vert = positif · ◀ rouge = négatif.
      </p>

      {data.map((d) => {
        const positif = d.contribution > 0
        const largeur = `${(Math.abs(d.contribution) / maxAbs) * 100}%`
        return (
          <div className="c-row" key={d.mot}>
            {/* Le mot */}
            <div className="c-mot">{d.mot}</div>

            {/* La piste divergente : moitié gauche (négatif) | moitié droite (positif) */}
            <div className="c-track">
              <div className="c-half c-left">
                {!positif && (
                  <div className="c-bar c-neg" style={{ width: largeur }} />
                )}
              </div>
              <div className="c-half c-right">
                {positif && (
                  <div className="c-bar c-pos" style={{ width: largeur }} />
                )}
              </div>
            </div>

            {/* La valeur chiffrée (signe = encodage redondant, lisible sans couleur) */}
            <div className={`c-val ${positif ? 'pos' : 'neg'}`}>
              {positif ? '+' : ''}{d.contribution.toFixed(2)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default Contributions
