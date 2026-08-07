
from contextlib import asynccontextmanager
 
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
from data_analysis import load_parquet
from optimization_problem import train_final_model, make_price_forecast, optimize_charging, naive_charging
 

# model and history are loaded/trained once at startup
model_state = {}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data and training model...")
    df = load_parquet("data/NO1_features.parquet")
    df = df[df["timestamp"] >= "2023-01-01"]
 
    best_params = pd.read_csv("data/best_lightgbm_params.csv").iloc[0].to_dict()
    best_params.update({"objective": "regression", "metric": "mae", "verbose": -1})
    for int_field in ["num_leaves", "n_estimators", "max_depth", "min_child_samples"]:
        best_params[int_field] = int(best_params[int_field])
 
    model_state["model"] = train_final_model(df, best_params)
    model_state["history"] = df
    print(f"Model ready. Last known hour: {df["timestamp"].max()}")
 
    yield
 
    model_state.clear()
 
 
app = FastAPI(
    title="Electricity Price Forecast API",
    description="Predicts NO1 electricity prices and suggests cost-optimal charging",
    lifespan=lifespan,
    )
 
 
class PricePoint(BaseModel):
    timestamp: str
    pris_nok_kwh: float
 
 
class ForecastResponse(BaseModel):
    generated_from: str
    prices: list[PricePoint]
 
 
class ChargingRecommendation(BaseModel):
    timestamp: str
    kwh: float
    pris_nok_kwh: float
 
 
class ChargingPlanResponse(BaseModel):
    total_kwh: float
    total_cost_kr: float
    naive_cost_kr: float
    savings_kr: float
    savings_pct: float
    plan: list[ChargingRecommendation]
 
 
class ChargingRequest(BaseModel):
    kwh_needed: float = Field(gt=0, description="Total energy amount to charge, in kWh")
    hours_ahead: int = Field(default=24, ge=1, le=72, description="How many hours ahead the deadline is")
    max_kwh_per_hour: float = Field(default=11.0, gt=0, description="Charger's power limit in kWh/hour")
 
 
@app.get("/")
def root():
    return {
        "name": "Electricity Price Forecast API",
        "endpoints": ["/forecast", "/charging-plan (POST)", "/health"],
        }
 
 
@app.get("/health")
def health():
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return {
        "status": "ok",
        "last_known_hour": str(model_state["history"]["timestamp"].max()),
        }
 
 
@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(hours_ahead: int = 24):
    """Return a price forecast for the next N hours (default: 24)."""
    
    if hours_ahead < 1 or hours_ahead > 72:
        raise HTTPException(status_code=400, detail="hours_ahead must be between 1 and 72")
 
    forecast = make_price_forecast(
        model_state["model"],
        model_state["history"],
        n_hours_ahead=hours_ahead,
        )
 
    return ForecastResponse(
        generated_from=str(model_state["history"]["timestamp"].max()),
        prices=[
            PricePoint(timestamp=time.isoformat(), pris_nok_kwh=round(price, 5))
            for time, price in zip(forecast["timestamp"], forecast["pris_nok_kwh"])
            ],
        )
 
 
@app.post("/charging-plan", response_model=ChargingPlanResponse)
def get_charging_plan(request: ChargingRequest):
    """Calculate a cost-optimal charging plan given an energy need and a deadline"""
    
    forecast = make_price_forecast(
        model_state["model"],
        model_state["history"],
        n_hours_ahead=request.hours_ahead,
        )
 
    try:
        plan = optimize_charging(
            forecast,
            kwh_needed=request.kwh_needed,
            max_kwh_per_hour=request.max_kwh_per_hour,
            )
        
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
 
    naive_cost = naive_charging(forecast, kwh_needed=request.kwh_needed)
    savings = naive_cost - plan.total_cost
 
    return ChargingPlanResponse(
        total_kwh=round(plan.total_kwh, 2),
        total_cost_kr=round(plan.total_cost, 2),
        naive_cost_kr=round(naive_cost, 2),
        savings_kr=round(savings, 2),
        savings_pct=round(savings / naive_cost * 100, 1) if naive_cost > 0 else 0.0,
        plan=[
            ChargingRecommendation(
                timestamp=time.isoformat(),
                kwh=round(kwh, 2),
                pris_nok_kwh=round(price, 5),
            )
            for time, kwh, price in zip(plan.hours, plan.kwh_per_hour, plan.prices)
            if kwh > 0.01
        ],
    )