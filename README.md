# Stratégie Alpha Cross-Sectionnel 130/30

## Setup

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

**Installer uv** (si pas déjà fait) :

macOS / Linux :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows :
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Installer les dépendances et lancer l'environnement** :
```bash
uv sync
```

C'est tout. `uv sync` crée le virtualenv et installe les dépendances épinglées dans `uv.lock`.