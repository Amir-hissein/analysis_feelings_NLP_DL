# ─────────────────────────────────────────────────────────────
# IMAGE DU FRONTEND — build multi-étapes (Phase 13)
# ─────────────────────────────────────────────────────────────

# ── Étape 1 : COMPILER le React (avec Node) ──────────────────
FROM node:22-alpine AS build
WORKDIR /app
# Installer les dépendances d'abord (cache si package.json inchangé).
COPY package*.json ./
RUN npm ci
# Compiler : npm run build -> dossier /app/dist (HTML/CSS/JS statiques optimisés).
COPY . .
RUN npm run build

# ── Étape 2 : SERVIR les fichiers statiques (avec nginx) ─────
# On NE garde QUE le résultat compilé : l'image finale ne contient
# pas Node ni node_modules -> beaucoup plus légère.
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
