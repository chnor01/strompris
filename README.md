# Strømpris-prognose

Prediker norske strømpriser (NO1) time for time med LightGBM, og bruk prognosen til å
optimalisere kostnaden ved strømforbruk (f.eks. elbil-lading) med lineær programmering.
Datainnsamling fra tre offentlige API-er (hvakosterstrommen.no, Frost/MET, NVE).
Applikasjonen kan testes via Streamlit-dashboardet.

## Oppsett

Repoet inkluderer et ferdig historisk datasett (`data/NO1_historical.parquet`),
så du kan hoppe rett til "Kjør lokalt" eller "Kjør med Docker" under uten å hente data selv.

Kjør `fetch_data.py` hvis du vil hente fersk data fra API-ene. Første innsamling kan ta en stund. Frost/MET API krever klient ID, se: https://frost.met.no/auth/requestCredentials.html

### (Valgfritt) Klient ID for Frost/MET API
```bash
cp .env.example .env
# fyll inn FROST_CLIENT_ID

uv sync
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
