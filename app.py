# app.py — Home / Dashboard
import os, time, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv

from common import (
    APP_NAME, FREE_DAYS, PREMIUM_DAYS, DEFAULT_BACKEND_URL,
    RECIPE_DB, Recipe,
    normalize_tokens, recipe_matches, get_day_slots,
    plan_to_dataframe, consolidate_shopping_list,
    pick_meals, pick_meals_ai,                 # keep both
    generate_ai_menu_with_recipes              # <-- include THIS here
)

# Planner fallback to ensure the symbol exists even if AI import fails
if "pick_meals_ai" not in globals():
    def pick_meals_ai(*args, **kwargs):
        return pick_meals(*args, **kwargs)

# --- Pantry helpers: safe import with fallbacks ---
try:
    from common import parse_pantry_text, split_shopping_by_pantry
except Exception:
    import re
    import pandas as _pd

    def _normalize_item_name(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s.endswith("es") and len(s) > 3:
            s = s[:-2]
        elif s.endswith("s") and len(s) > 3:
            s = s[:-1]
        return s

    def parse_pantry_text(text: str) -> list[str]:
        if not text:
            return []
        raw = re.split(r"[,\n]", text)
        return [i.strip() for i in raw if i.strip()]

    def split_shopping_by_pantry(df_shop: _pd.DataFrame, pantry_items: list[str], annotate_at_bottom: bool = False):
        if df_shop is None or df_shop.empty:
            return df_shop, _pd.DataFrame(columns=df_shop.columns if df_shop is not None else ["item","quantity","unit"])

        norm_map = {idx: _normalize_item_name(str(row["item"])) for idx, row in df_shop.iterrows()}
        pantry_norm = [_normalize_item_name(p) for p in pantry_items]

        need_rows, have_rows = [], []
        for idx, row in df_shop.iterrows():
            norm_item = norm_map[idx]
            matched = any(p and (p in norm_item or norm_item in p) for p in pantry_norm)
            if matched:
                have_rows.append(row)
                if annotate_at_bottom:
                    need_rows.append(row)  # keep it in main list too (we'll annotate)
            else:
                need_rows.append(row)

        need_df = _pd.DataFrame(need_rows).reset_index(drop=True)
        have_df = _pd.DataFrame(have_rows).reset_index(drop=True)
        return need_df, have_df
# --- end pantry helpers fallback ---

load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon="🥗", layout="wide")

# --- sanity check: is the OpenRouter key visible? (don’t print the key) ---
if st.session_state.get("use_ai_toggle", False):  # only show when AI toggle is on
    if not (st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        st.error("OPENROUTER_API_KEY not found in secrets/env. Add it in App → Settings → Secrets and rerun.")

# --- Warm the backend so it wakes up before we need it ---
try:
    requests.get(f"{DEFAULT_BACKEND_URL}/health", timeout=8)
except Exception:
    # it's fine if this fails; verification has its own retries
    pass

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

st.sidebar.markdown("### Pantry (optional)")
pantry_text = st.sidebar.text_area(
    "Items you already have (comma or new line separated)",
    value=st.session_state.get("pantry_text", ""),
    placeholder="rice, olive oil\nbananas",
    height=120,
)
st.session_state["pantry_text"] = pantry_text
show_pantry_note = st.sidebar.checkbox(
    "Keep pantry items in list (annotate instead of removing)",
    value=False,
    help="If checked, pantry items also appear in the main list so you can mark them '(have)'",
)
st.session_state["show_pantry_note"] = show_pantry_note

# ---- Filter recipes ----
filtered = [r for r in RECIPE_DB if recipe_matches(
    r, diet_flags, normalize_tokens(allergies), normalize_tokens(exclusions), cuisines
)]

if not filtered:
    st.warning("No recipes match your filters. Try removing some restrictions 🤔")
    st.stop()

# ---- Plan generation (manual + lock + save/load) ----
import json
import hashlib

st.subheader(f"Your {days}-day plan")

# AI toggle still Premium-only
use_ai = st.session_state.is_premium and st.toggle("Use AI to draft plan", value=True, key="use_ai_toggle")

ai_mode = "Pick from built‑in"
if use_ai:
    ai_mode = st.radio(
        "AI mode",
        ["Pick from built‑in", "Generate new recipes"],
        index=0,
        horizontal=True,
        help="Pick from your recipe database, or ask AI to create new recipes with ingredients & steps.",
    )

# Create a signature of “inputs that define a plan”
def make_filters_signature() -> str:
    parts = [
        tuple(sorted(diet_flags)),
        tuple(sorted(cuisines)),
        tuple(sorted(normalize_tokens(allergies))),
        tuple(sorted(normalize_tokens(exclusions))),
        meals_per_day,
        days,
        int(st.session_state.calorie_target) if st.session_state.is_premium else None,
        bool(use_ai),
    ]
    raw = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

sig = make_filters_signature()

with st.expander("AI connection test (optional)"):
    if st.button("Run a 1-shot OpenRouter test"):
        try:
            from common import call_openrouter, OPENROUTER_MODEL
            resp = call_openrouter(
                [{"role": "user", "content": "Reply with the word OK"}],
                model=OPENROUTER_MODEL,
                max_tokens=5,
            )
            st.success("OpenRouter call succeeded.")
            st.write(resp.get("choices", [{}])[0].get("message", {}))
        except Exception as e:
            st.error(f"OpenRouter call failed: {e}")

# Controls
cols = st.columns([0.35, 0.35, 0.30])
with cols[0]:
    gen_clicked = st.button("🔁 Generate / Regenerate plan", type="primary", use_container_width=True)
with cols[1]:
    st.session_state.plan_locked = st.checkbox(
        "🔒 Lock this plan (don’t auto-change)",
        value=st.session_state.get("plan_locked", False)
    )
with cols[2]:
    # Save/Load plan JSON
    dl_plan = None
    if "plan" in st.session_state:
        dl_plan = json.dumps(st.session_state.plan, ensure_ascii=False, indent=0).encode()
        st.download_button(
            "⬇️ Download plan (JSON)",
            data=dl_plan,
            file_name="mealplan.json",
            mime="application/json",
            use_container_width=True
        )

# Load JSON
uploaded = st.file_uploader("⬆️ Load plan (JSON)", type=["json"], label_visibility="collapsed")
if uploaded is not None:
    try:
        loaded_plan = json.loads(uploaded.read().decode("utf-8"))
        # basic validation: expect dict of day->list
        if isinstance(loaded_plan, dict):
            st.session_state.plan = loaded_plan
            st.session_state.filters_sig = sig  # assume this plan corresponds to current filters
            st.success("Plan loaded from file.")
        else:
            st.error("Invalid plan file format.")
    except Exception as e:
        st.error(f"Could not load plan: {e}")

# Decide if we should (re)generate
should_generate = False
existing_plan = st.session_state.get("plan")

if not existing_plan:
    should_generate = True
elif gen_clicked:
    should_generate = True
elif not st.session_state.get("plan_locked", False) and st.session_state.get("filters_sig") != sig:
    should_generate = True
elif st.session_state.get("plan_locked", False) and st.session_state.get("filters_sig") != sig:
    st.info("Your filters changed, but the plan is locked. Click Generate or Regenerate to update.")

if should_generate:
    if use_ai and ai_mode == "Generate new recipes":
        ai_plan = generate_ai_menu_with_recipes(
            days=days,
            meals_per_day=meals_per_day,
            diets=list(diet_flags),
            allergies=normalize_tokens(allergies),
            exclusions=normalize_tokens(exclusions),
            cuisines=list(cuisines),
            calorie_target=st.session_state.calorie_target if st.session_state.is_premium else None,
        )
        if ai_plan:
            st.session_state.plan = ai_plan
        else:
            # <- DO NOT silently continue; make it obvious
            st.session_state.plan = {}
            st.stop()  # shows the error from common.py and stops rendering
    else:
        generator = pick_meals_ai if use_ai else pick_meals
        st.session_state.plan = generator(
            filtered, meals_per_day, days,
            st.session_state.calorie_target if st.session_state.is_premium else None
        )
    st.session_state.filters_sig = sig

# Use the plan from session
plan = st.session_state.plan
df_plan = plan_to_dataframe(plan, meals_per_day)

# ---- View renderer (single-file, robust) ----
st.markdown("---")
plan = st.session_state.plan

from common import get_day_slots, plan_to_dataframe, consolidate_shopping_list
from recipe_db import RECIPE_DB

if view == "Today":
    st.subheader("📅 Today’s Meals")

    if not plan or not list(plan.keys()):
        st.info("No plan yet. Click **Generate / Regenerate plan** above.")
        st.stop()

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

    # Build dataframes safely
    df_plan2 = plan_to_dataframe(plan, meals_per_day) if plan else pd.DataFrame()

    # If there’s nothing to show yet, stop before grouping
    if df_plan2 is None or df_plan2.empty or "day" not in df_plan2.columns:
        st.info("No plan data to summarize. Click **Generate / Regenerate plan** first.")
        st.stop()

    df_shop2 = consolidate_shopping_list(plan)

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.dataframe(df_plan2, use_container_width=True, hide_index=True)

        # Daily totals only if columns are present
        needed_cols = {"calories", "protein_g", "carbs_g", "fat_g"}
        if st.session_state.is_premium and needed_cols.issubset(set(df_plan2.columns)):
            day_summary = (
                df_plan2.groupby("day")[["calories", "protein_g", "carbs_g", "fat_g"]]
                .sum()
                .reset_index()
            )
            st.markdown("**Daily totals**")
            st.dataframe(day_summary, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Shopping list**")

        # Pantry split (re-use your helpers)
        pantry_items = parse_pantry_text(st.session_state.get("pantry_text", ""))
        annotate = st.session_state.get("show_pantry_note", False)
        need_df, have_df = split_shopping_by_pantry(df_shop2, pantry_items, annotate_at_bottom=annotate)

        if annotate:
            if not need_df.empty:
                need_df_display = need_df.copy()
                norm_have = {i.lower() for i in have_df["item"].astype(str)} if not have_df.empty else set()
                need_df_display["item"] = need_df_display["item"].astype(str).apply(
                    lambda x: f"{x} (have)" if x.lower() in norm_have else x
                )
                st.dataframe(need_df_display, use_container_width=True, hide_index=True)
            else:
                st.info("No items needed.")
        else:
            st.markdown("**Items to buy**")
            st.dataframe(need_df, use_container_width=True, hide_index=True)
            with st.expander("Pantry items (matched)"):
                if have_df.empty:
                    st.write("No matches.")
                else:
                    st.dataframe(have_df, use_container_width=True, hide_index=True)

    # Downloads
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
        st.download_button(
            "Download Shopping List (CSV)",
            data=df_shop2.to_csv(index=False).encode(),
            file_name="shopping_list.csv",
            mime="text/csv",
        )



elif view == "Recipes":
    st.subheader("📖 Recipes in this plan")

    # Pull recipes directly from the plan (works for DB-picked & AI-generated)
    used = [r for meals in plan.values() for r in meals if r]

    q = st.text_input("Search recipe name or ingredient", "")
    def matches(r):
        if not q:
            return True
        ql = q.lower()
        if ql in (r.get("name","").lower()):
            return True
        return any(ql in str(ing.get("item","")).lower() for ing in r.get("ingredients", []))

    for r in [r for r in used if matches(r)]:
        with st.expander(r.get("name","Recipe")):
            st.write(f"*Cuisine:* `{r.get('cuisine','')}` — *Course:* `{r.get('course','any')}`")
            st.write("**Ingredients:**")
            for ing in r.get("ingredients", []):
                st.write(f"- {ing.get('qty','')} {ing.get('unit','')} {ing.get('item','')}".strip())
            st.write("**Steps:**")
            for i, step in enumerate(r.get("steps", []), start=1):
                st.write(f"{i}. {step}")
