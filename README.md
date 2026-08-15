# Strømpris-prognose

Prediker norske strømpriser (NO1) time for time med LightGBM, og bruk prognosen til å
optimalisere kostnaden ved strømforbruk (f.eks. elbil-lading) med lineær programmering.
Datainnsamling fra tre offentlige API-er (hvakosterstrommen.no, Frost/MET, NVE).
Applikasjonen kan testes via Streamlit-dashboardet.

## Oppsett

Repoet inkluderer et ferdig historisk datasett (`data/NO1_historical.parquet`),
så du kan hoppe rett til "Kjør lokalt" eller "Kjør med Docker" under uten å hente data selv.

### (Valgfritt) Klient ID for Frost/MET API
```bash
cp .env.example .env
# fyll inn FROST_CLIENT_ID (gratis, registrer via https://frost.met.no/auth/requestCredentials.html)

uv sync
```

### (Valgfritt) Hent nyere data selv

```bash
uv run python -c "from fetch_data import fetch_data_range; fetch_data_range()"
```

### Kjør lokalt

```bash
uv run uvicorn api:app --reload        # terminal 1
uv run streamlit run dashboard.py      # terminal 2
```

### Kjør med Docker

```bash
docker compose up --build
```

API: `http://localhost:8000/docs` Dashboard: `http://localhost:8501`
