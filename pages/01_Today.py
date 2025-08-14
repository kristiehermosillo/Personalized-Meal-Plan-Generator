# pages/01_Today.py
import streamlit as st
from common import get_day_slots

st.set_page_config(page_title="Today", page_icon="📅", layout="wide")
st.title("📅 Today’s Meals")

plan = st.session_state.get("plan", {})

# Guard: stop politely if no plan
if not plan or not list(plan.keys()):
    st.info("No plan yet. Go back to **Home** and click **Generate / Regenerate plan**.")
    st.stop()

# Determine number of days
max_day = max(plan.keys())

# Horizontal day picker
day_label = st.radio(
    "Pick a day",
    [f"Day {i}" for i in range(1, max_day + 1)],
    index=0,
    horizontal=True,
)
day = int(day_label.split()[1])

# Slots from first day (fallback to 3 if not present)
meals_today = plan.get(day, [])
meals_per_day = len(plan.get(1, [])) or 3
slots = get_day_slots(meals_per_day)

for i, r in enumerate(meals_today, start=1):
    label = slots[i - 1] if i - 1 < len(slots) else f"Meal {i}"
    if not r:
        st.write(f"**{label}** — (empty)")
        continue

    with st.expander(f"{label} — {r.get('name','Recipe')}", expanded=False):
        st.write("**Ingredients:**")
        for ing in r.get("ingredients", []):
            qty = ing.get("qty", "")
            unit = ing.get("unit", "")
            item = ing.get("item", "")
            line = f"- {qty} {unit} {item}".strip()
            st.write(line)
        st.write("**Steps:**")
        for idx, step in enumerate(r.get("steps", []), start=1):
            st.write(f"{idx}. {step}")
