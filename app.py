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
    """If redirected back from Stripe with session_id, verify it (with retries)."""
    sess_id = get_session_id_from_url()
    if not sess_id or sess_id == st.session_state.get("last_session_id"):
        return
    import time
    last_err = None
    for _ in range(3):  # 3 tries
        try:
            r = requests.get(
                f"{DEFAULT_BACKEND_URL}/verify-session",
                params={"session_id": sess_id},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("paid"):
                st.session_state.is_premium = True
                st.session_state.last_session_id = sess_id
                st.success("✅ Premium unlocked! Enjoy the full features.")
                return
            else:
                st.info("Payment not completed yet.")
                return
        except Exception as e:
            last_err = e
            time.sleep(2)
    st.warning(f"Could not verify payment (server waking up). Please try again. Details: {last_err}")
    # Optional manual retry button
    if st.button("🔁 Retry verification", use_container_width=True):
        st.rerun()

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

st.sidebar.markdown("### Navigation")
view = st.sidebar.radio(
    "Go to",
    ["Today", "Weekly Overview", "Recipes"],
    index=0,
    horizontal=False,
)

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

# ---- View renderer (single-file, robust) ----
st.markdown("---")
plan = st.session_state.plan
meals_per_day = len(plan[1]) if plan.get(1) else 3

from common import get_day_slots, plan_to_dataframe, consolidate_shopping_list
from recipe_db import RECIPE_DB

if view == "Today":
    st.subheader("📅 Today’s Meals")

    max_day = max(plan.keys())

    # Horizontal day picker (no slider)
    day_label = st.radio(
        "Pick a day",
        [f"Day {i}" for i in range(1, max_day + 1)],
        index=0,
        horizontal=True,
    )
    day = int(day_label.split()[1])

    slots = get_day_slots(meals_per_day)
    meals = plan.get(day, [])

    for i, r in enumerate(meals, start=1):
        label = slots[i - 1] if i - 1 < len(slots) else f"Meal {i}"
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

elif view == "Weekly Overview":
    st.subheader("🗓️ Weekly Overview")

    df_plan2 = plan_to_dataframe(plan, meals_per_day)
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.dataframe(df_plan2, use_container_width=True, hide_index=True)
        if st.session_state.is_premium:
            day_summary = (
                df_plan2.groupby("day")[["calories", "protein_g", "carbs_g", "fat_g"]]
                .sum()
                .reset_index()
            )
            st.markdown("**Daily totals**")
            st.dataframe(day_summary, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Shopping list**")
        df_shop2 = consolidate_shopping_list(plan)
        st.dataframe(df_shop2, use_container_width=True, hide_index=True)

    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Plan (CSV)",
            data=df_plan2.to_csv(index=False).encode(),
            file_name="mealplan.csv",
            mime="text/csv",
        )
    with dl2:
        df_shop2 = consolidate_shopping_list(plan)
        st.download_button(
            "Download Shopping List (CSV)",
            data=df_shop2.to_csv(index=False).encode(),
            file_name="shopping_list.csv",
            mime="text/csv",
        )

elif view == "Recipes":
    st.subheader("📖 Recipes in this plan")

    used_names = {r["name"] for meals in plan.values() for r in meals if r}
    used = [r for r in RECIPE_DB if r["name"] in used_names]

    q = st.text_input("Search recipe name or ingredient", "")
    def matches(r):
        if not q:
            return True
        ql = q.lower()
        if ql in r["name"].lower():
            return True
        return any(ql in ing["item"].lower() for ing in r["ingredients"])

    for r in [r for r in used if matches(r)]:
        with st.expander(r["name"]):
            st.write(f"*Cuisine:* `{r.get('cuisine','')}` — *Course:* `{r.get('course','any')}`")
            st.write("**Ingredients:**")
            for ing in r["ingredients"]:
                st.write(f"- {ing.get('qty','')} {ing.get('unit','')} {ing['item']}".strip())
            st.write("**Steps:**")
            for i, step in enumerate(r.get("steps", []), start=1):
                st.write(f"{i}. {step}")
