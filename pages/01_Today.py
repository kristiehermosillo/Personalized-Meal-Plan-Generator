# pages/01_Today.py
import streamlit as st
from common import ensure_plan_exists, get_day_slots, plan_to_dataframe

st.set_page_config(page_title="Today's Meals — MealPlan Genie", page_icon="📅", layout="wide")
st.title("📅 Today’s Meals")

ensure_plan_exists()
plan = st.session_state.plan

# Choose which day to view
max_day = max(plan.keys())
day = st.slider("Select day", 1, max_day, 1)
slots = get_day_slots(len(plan[1]))

st.subheader(f"Day {day}")
meals = plan.get(day, [])
for i, r in enumerate(meals, start=1):
    label = slots[i-1] if i-1 < len(slots) else f"Meal {i}"
    if not r:
        st.write(f"**{label}** — (empty)")
        continue
    with st.expander(f"{label} — {r['name']}", expanded=False):
        st.write("**Ingredients:**")
        for ing in r["ingredients"]:
            st.write(f"- {ing.get('qty','')} {ing.get('unit','')} {ing['item']}".strip())
        st.write("**Steps:**")
        for idx, step in enumerate(r.get("steps", []), start=1):
            st.write(f"{idx}. {step}")

st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=True):
    st.switch_page("app.py")
