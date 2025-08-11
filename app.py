# app.py  — Home / Dashboard
import os, time, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
from common import (
    APP_NAME, FREE_DAYS, PREMIUM_DAYS, DEFAULT_BACKEND_URL,
    RECIPE_DB, Recipe,
    normalize_tokens, recipe_matches, get_day_slots,
    pick_meals, pick_meals_ai, plan_to_dataframe
)

load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon="🥗", layout="wide")

# ---- Session bootstrap ----
for k, v in {
    "is_premium": False, "calorie_target": 2000,
    "last_session_id": None, "used_ai_prev": False
}.items():
    st.session_state.setdefault(k, v)

# ---- Verify Stripe session if redirected back ----
def get_session_id_from_url():
    try:
        qp = dict(st.query_params); s = qp.get("session_id"); 
        return s[0] if isinstance(s,list) else s
    except: return None

def check_stripe_session():
    sid = get_session_id_from_url()
    if not sid or sid == st.session_state.get("last_session_id"): return
    try:
        r = requests.get(f"{DEFAULT_BACKEND_URL}/verify-session", params={"session_id": sid}, timeout=30)
        r.raise_for_status(); data = r.json()
        if data.get("paid"): 
            st.session_state.is_premium = True
            st.session_state.last_session_id = sid
            st.success("✅ Premium unlocked! Enjoy the full features.")
        else:
            st.info("Payment not completed yet.")
    except Exception as e:
        st.warning(f"Could not verify payment: {e}")

check_stripe_session()

# ---- Header ----
left, right = st.columns([0.70, 0.30])
with left:
    st.title("🥗 MealPlan Genie")
    st.caption("Fast, tasty weekly plans tailored to your diet. Free 3‑day preview. Upgrade for full power.")
with right:
    if st.session_state.is_premium:
        st.success("Premium ✅")
    else:
        with st.popover("Premium 🔓"):
            st.markdown("**Unlock Premium** to get:\n- 7‑day plan\n- Calorie targeting\n- Macros per day & meal\n- PDF export")
            if st.button("Upgrade with Stripe", use_container_width=True, type="primary"):
                url = None
                for tries in range(3):
                    try:
                        r = requests.post(f"{DEFAULT_BACKEND_URL}/create-checkout-session", timeout=45)
                        r.raise_for_status(); url = r.json().get("checkout_url"); break
                    except Exception as e:
                        if tries == 2: st.error(f"Failed to create checkout session: {e}"); break
                        time.sleep(3)
                if url: st.markdown(f"[Click to open Stripe Checkout]({url})")

st.divider()

# ---- Sidebar inputs ----
st.sidebar.header("Your Preferences")
diet_flags = st.sidebar.multiselect("Dietary style",
    ["vegetarian","vegan","gluten-free","dairy-free","pescatarian","low-carb"], default=[])
allergies  = st.sidebar.text_input("Allergies (comma-separated)", "")
exclusions = st.sidebar.text_input("Disliked ingredients (comma-separated)", "")
meals_per_day = st.sidebar.slider("Meals per day", 2, 4, 3)
if st.session_state.is_premium:
    st.session_state.calorie_target = st.sidebar.slider("Daily calorie target", 1200, 3200, st.session_state.calorie_target, step=100)
else:
    st.sidebar.info("Calorie targeting available in Premium.")
cuisines = st.sidebar.multiselect("Cuisine preference (optional)",
    ["american","mediterranean","asian","mexican","indian","middle-eastern","italian"], default=[])

days = PREMIUM_DAYS if st.session_state.is_premium else FREE_DAYS
st.sidebar.caption("Free: 3‑day plan preview. Upgrade for 7 days + macros + PDF export.")

# ---- Filter recipes ----
filtered = [r for r in RECIPE_DB if recipe_matches(
    r, diet_flags, normalize_tokens(allergies), normalize_tokens(exclusions), cuisines
)]

if not filtered:
    st.warning("No recipes match your filters. Try removing some restrictions 🤔")
    st.stop()

# ---- Plan generation (with AI toggle) ----
st.subheader(f"Your {days}-day plan")
use_ai = st.session_state.is_premium and st.toggle("Use AI to draft plan", value=True)

regen_needed = (
    "plan" not in st.session_state
    or st.session_state.get("meals_per_day_prev") != meals_per_day
    or st.session_state.get("days_prev") != days
    or st.session_state.get("used_ai_prev") != bool(use_ai)
    or st.session_state.get("filters_hash") != hash(tuple(sorted([*diet_flags,*cuisines,*normalize_tokens(allergies),*normalize_tokens(exclusions)])))
)

if regen_needed:
    generator = pick_meals_ai if use_ai else pick_meals
    st.session_state.plan = generator(
        filtered, meals_per_day, days,
        st.session_state.calorie_target if st.session_state.is_premium else None
    )
    st.session_state.meals_per_day_prev = meals_per_day
    st.session_state.days_prev = days
    st.session_state.used_ai_prev = bool(use_ai)
    st.session_state.filters_hash = hash(tuple(sorted([*diet_flags,*cuisines,*normalize_tokens(allergies),*normalize_tokens(exclusions)])))

plan = st.session_state.plan
df_plan = plan_to_dataframe(plan, meals_per_day)

# Quick peek (Home keeps it simple)
st.dataframe(df_plan, use_container_width=True, hide_index=True)
st.info("Use the pages in the left sidebar for **Today**, **Weekly Overview**, and **Recipes**.")
