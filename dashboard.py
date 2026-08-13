import pandas as pd, httpx, streamlit as st, os
 
API_URL = os.environ.get("API_URL", "http://localhost:8000")
 
st.set_page_config(page_title="Electricity Price Forecast", layout="wide")
 
st.title("Electricity Price Forecast & Load Scheduling")
st.caption("NO1 electricity prices predicted with LightGBM, load scheduling optimized with linear programming")
 
 
@st.cache_data(ttl=300)
def get_health():
    try:
        res = httpx.get(f"{API_URL}/health", timeout=5)
        res.raise_for_status()
        return res.json()
    except httpx.HTTPError:
        return None

 
@st.cache_data(ttl=300)
def get_forecast(hours_ahead: int):
    res = httpx.get(f"{API_URL}/forecast", params={"hours_ahead": hours_ahead}, timeout=30)
    res.raise_for_status()
    return res.json()
 
 
def get_charging_plan(kwh_needed: float, hours_ahead: int, max_kwh_per_hour: float):
    res = httpx.post(
        f"{API_URL}/charging-plan",
        json={
            "kwh_needed": kwh_needed,
            "hours_ahead": hours_ahead,
            "max_kwh_per_hour": max_kwh_per_hour,
            },
        timeout=30,
        )
    res.raise_for_status()
    return res.json()
 
 
health = get_health()
 
if health is None:
    st.error("Could not reach the API. Make sure it's running")
    st.stop()
 
st.success(f"Model ready - last known hour: {health["last_known_hour"]}")
 
st.sidebar.header("Charging Schedule")
kwh_needed = st.sidebar.number_input("Energy needed (kWh)", min_value=1.0, max_value=200.0, value=40.0, step=1.0)
hours_ahead = st.sidebar.slider("Deadline (hours ahead)", min_value=1, max_value=72, value=24)
max_kwh_per_hour = st.sidebar.number_input("Max charging power (kWh/hour)", min_value=1.0, max_value=50.0, value=11.0, step=0.5)
compute = st.sidebar.button("Compute charging plan", type="primary")
 
col1, col2 = st.columns([2, 1])
 
with col1:
    st.subheader("Price Forecast")
    forecast_data = get_forecast(hours_ahead)
    forecast_df = pd.DataFrame(forecast_data["prices"])
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
 
    st.line_chart(forecast_df.set_index("timestamp")["pris_nok_kwh"])
 
with col2:
    st.subheader("Key Figures")
    st.metric("Lowest price in period", f"{forecast_df["pris_nok_kwh"].min():.3f} kr/kWh")
    st.metric("Highest price in period", f"{forecast_df["pris_nok_kwh"].max():.3f} kr/kWh")
    st.metric("Average price", f"{forecast_df["pris_nok_kwh"].mean():.3f} kr/kWh")
 
st.divider()
 
if compute:
    st.subheader("Recommended Charging Plan")
 
    try:
        plan = get_charging_plan(kwh_needed, hours_ahead, max_kwh_per_hour)
    except httpx.HTTPError as exc:
        st.error(f"Could not compute charging plan: {exc}")
        st.stop()
 
    m1, m2, m3 = st.columns(3)
    m1.metric("Cost (optimized)", f"{plan["total_cost_kr"]:.2f} kr")
    m2.metric("Cost (naive, flat charging)", f"{plan["naive_cost_kr"]:.2f} kr")
    m3.metric("Savings", f"{plan["savings_kr"]:.2f} kr", delta=f"{plan["savings_pct"]:.1f}%")
 
    plan_df = pd.DataFrame(plan["plan"])
    
    if not plan_df.empty:
        plan_df["timestamp"] = pd.to_datetime(plan_df["timestamp"])
        plan_df = plan_df.sort_values("timestamp").reset_index(drop=True)
        plan_df["timestamp"] = plan_df["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(
            plan_df.rename(columns={
                "timestamp": "Time",
                "kwh": "kWh",
                "pris_nok_kwh": "Price (kr/kWh)"
                }),
            hide_index=True,
            )
    else:
        st.info("No charging sessions in the plan.")
        
else:
    st.info("Set energy needed and deadline in the sidebar, then click 'Compute charging plan'.")