# app.py
import os
import urllib.parse as up
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

from recipe_db import RECIPE_DB, Recipe

from dotenv import load_dotenv
import os

load_dotenv()  # this loads the .env file

APP_NAME = "MealPlan Genie"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PREMIUM_FEATURES = [
    "7‑day plan",
    "Calorie target matching",
    "Macros per meal & per day",
    "PDF export",
]

FREE_DAYS = 3
PREMIUM_DAYS = 7

st.set_page_config(page_title=APP_NAME, page_icon="🥗", layout="wide")

# --- Session bootstrap ---
if "is_premium" not in st.session_state:
    st.session_state.is_premium = False
if "calorie_target" not in st.session_state:
    st.session_state.calorie_target = 2000
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None

# --- Query params (new API only) ---
def get_session_id_from_url():
    try:
        # Streamlit 1.32+ : st.query_params is dict-like
        qp = dict(st.query_params)
        sess_id = qp.get("session_id")
        if isinstance(sess_id, list):
            sess_id = sess_id[0]
        return sess_id
    except Exception:
        return None

def check_stripe_session():
    """If redirected back from Stripe with session_id, verify it."""
    sess_id = get_session_id_from_url()
    if not sess_id or sess_id == st.session_state.get("last_session_id"):
        return
    try:
        r = requests.get(f"{DEFAULT_BACKEND_URL}/verify-session",
                         params={"session_id": sess_id}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("paid") is True:
            st.session_state.is_premium = True
            st.session_state.last_session_id = sess_id
            st.success("✅ Premium unlocked! Enjoy the full features.")
        else:
            st.info("Payment not completed yet.")
    except Exception as e:
        st.warning(f"Could not verify payment: {e}")

check_stripe_session()

# --- UI: Header ---
left, right = st.columns([0.75, 0.25])
with left:
    st.title("🥗 MealPlan Genie")
    st.caption("Fast, tasty weekly plans tailored to your diet. Free 3‑day preview. Upgrade for full power.")
with right:
    if st.session_state.is_premium:
        st.success("Premium ✅")
    else:
        with st.popover("Premium 🔓"):
            st.markdown("**Unlock Premium** to get:")
            for f in PREMIUM_FEATURES:
                st.markdown(f"- {f}")
            if st.button("Upgrade with Stripe", use_container_width=True, type="primary"):
                try:
                    r = requests.post(f"{DEFAULT_BACKEND_URL}/create-checkout-session", timeout=10)
                    r.raise_for_status()
                    url = r.json().get("checkout_url")
                    if url:
                        st.markdown(f"[Click to open Stripe Checkout]({url})")
                    else:
                        st.error("No checkout URL returned.")
                except Exception as e:
                    st.error(f"Failed to create checkout session: {e}")

st.divider()

# --- Sidebar: Inputs ---
st.sidebar.header("Your Preferences")
diet_flags = st.sidebar.multiselect(
    "Dietary style",
    ["vegetarian", "vegan", "gluten-free", "dairy-free", "pescatarian", "low-carb"],
    default=[]
)
allergies = st.sidebar.text_input("Allergies (comma-separated)", value="")
exclusions = st.sidebar.text_input("Disliked ingredients (comma-separated)", value="")
meals_per_day = st.sidebar.slider("Meals per day", 2, 4, 3)

if st.session_state.is_premium:
    cal_target = st.sidebar.slider("Daily calorie target", 1200, 3200, st.session_state.calorie_target, step=100)
    st.session_state.calorie_target = cal_target
else:
    st.sidebar.info("Calorie targeting available in Premium.")

cuisines = st.sidebar.multiselect(
    "Cuisine preference (optional)",
    ["american", "mediterranean", "asian", "mexican", "indian", "middle-eastern", "italian"],
    default=[]
)

days = PREMIUM_DAYS if st.session_state.is_premium else FREE_DAYS

st.sidebar.markdown("---")
if not st.session_state.is_premium:
    st.sidebar.caption("Free: 3‑day plan preview. Upgrade for 7 days + macros + PDF export.")

# --- Core Logic ---
def get_day_slots(meals_per_day: int) -> list[str]:
    if meals_per_day <= 2:
        return ["Breakfast", "Dinner"]
    elif meals_per_day == 3:
        return ["Breakfast", "Lunch", "Dinner"]
    else:  # 4 meals
        return ["Breakfast", "Lunch", "Dinner", "Snack"]

def normalize_tokens(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def recipe_matches(recipe: Recipe, diets: List[str], allergies: List[str], exclusions: List[str], cuisines: List[str]) -> bool:
    # Diet tags must include all selected diet flags
    for d in diets:
        if d not in recipe["diet_tags"]:
            return False
    # Allergies/exclusions: reject if any present in ingredients
    ing_tokens = [i["item"].lower() for i in recipe["ingredients"]]
    for bad in allergies + exclusions:
        if any(bad in tok for tok in ing_tokens):
            return False
    # Cuisine filter if provided
    if cuisines:
        if recipe.get("cuisine") not in cuisines:
            return False
    return True

def pick_meals(filtered: List[Recipe], meals_per_day: int, days: int, cal_target: int | None) -> Dict[int, List[Recipe]]:
    """
    Pick meals by correct course:
    - Breakfast slot -> 'breakfast' (or 'any')
    - Lunch slot     -> 'lunch' (or 'any')
    - Dinner slot    -> 'dinner' (or 'any')
    - Snack slot     -> 'any' (fallback to anything if empty)
    Avoid repeats within a day. If cal_target set, nudge daily total toward it.
    """
    import random
    random.seed(42)

    # Buckets by course
    buckets: Dict[str, List[Recipe]] = {"breakfast": [], "lunch": [], "dinner": [], "any": []}
    for r in filtered:
        buckets.setdefault(r.get("course", "any"), buckets["any"]).append(r)

    # Slot keys (for filtering) and display names
    slot_keys = []
    for name in get_day_slots(meals_per_day):
        if name == "Breakfast": slot_keys.append("breakfast")
        elif name == "Lunch":   slot_keys.append("lunch")
        elif name == "Dinner":  slot_keys.append("dinner")
        else:                   slot_keys.append("any")  # Snack

    def candidates_for(slot_key: str) -> List[Recipe]:
        # Prefer exact course, then 'any'
        names = set()
        out = []
        for r in buckets.get(slot_key, []):
            if r["name"] not in names:
                out.append(r); names.add(r["name"])
        for r in buckets.get("any", []):
            if r["name"] not in names:
                out.append(r); names.add(r["name"])
        return out

    plan: Dict[int, List[Recipe]] = {}
    for d in range(1, days + 1):
        used_today = set()
        meals_today: List[Recipe] = []
        current_cals = 0

        for i, slot_key in enumerate(slot_keys, start=1):
            cands = [r for r in candidates_for(slot_key) if r["name"] not in used_today]
            if not cands:
                cands = [r for r in filtered if r["name"] not in used_today]
            if not cands:
                continue

            if cal_target:
                remaining_slots = len(slot_keys) - (i - 1)
                desired = max(180, int((cal_target - current_cals) / max(1, remaining_slots)))
                cands.sort(key=lambda r: abs(r["calories"] - desired))
                choice = cands[0]
            else:
                random.shuffle(cands)
                choice = cands[0]

            meals_today.append(choice)
            used_today.add(choice["name"])
            current_cals += choice["calories"]

        plan[d] = meals_today

    return plan

    # Category buckets by course to diversify
    buckets: Dict[str, List[Recipe]] = {"breakfast": [], "lunch": [], "dinner": [], "any": []}
    for r in pool:
        buckets.get(r.get("course", "any"), buckets["any"]).append(r)

    def next_meal(prev_used: set) -> Recipe | None:
        # try rotating through buckets
        for key in ["breakfast", "lunch", "dinner", "any"]:
            random.shuffle(buckets[key])
            for r in buckets[key]:
                if r["name"] in prev_used:
                    continue
                return r
        return None

    for d in range(1, days + 1):
        used = set()
        meals: List[Recipe] = []
        attempts = 0
        while len(meals) < meals_per_day and attempts < 200:
            r = next_meal(used)
            if not r:
                break
            if cal_target and len(meals) == meals_per_day - 1:
                # try to choose a last meal that nudges calories toward target
                current = sum(m["calories"] for m in meals)
                target_remaining = cal_target - current
                # find candidate closest to target_remaining / remaining_meals
                remaining = meals_per_day - len(meals)
                desired = max(200, int(target_remaining / remaining))
                candidates = sorted(pool, key=lambda x: abs(x["calories"] - desired))
                for c in candidates:
                    if c["name"] not in used:
                        r = c
                        break
            meals.append(r)
            used.add(r["name"])
            attempts += 1

        # Premium: ensure total within band, otherwise simple greedy adjust
        if cal_target:
            total = sum(m["calories"] for m in meals)
            band_low, band_high = int(cal_target * 0.88), int(cal_target * 1.12)
            tweak_rounds = 0
            while (total < band_low or total > band_high) and tweak_rounds < 50:
                # try replacing a random meal with a closer calorie option
                idx = random.randrange(0, len(meals))
                current = meals[idx]
                diff = cal_target - total
                desired = max(180, current["calories"] + diff // 2)
                candidates = sorted(pool, key=lambda x: abs(x["calories"] - desired))
                for c in candidates:
                    if c["name"] not in used:
                        used.remove(current["name"])
                        meals[idx] = c
                        used.add(c["name"])
                        break
                total = sum(m["calories"] for m in meals)
                tweak_rounds += 1

        day_plan[d] = meals
    return day_plan

def consolidate_shopping_list(plan: Dict[int, List[Recipe]]) -> pd.DataFrame:
    from collections import defaultdict
    totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for meals in plan.values():
        for rec in meals:
            if not rec:  # skip empty slots
                continue
            for ing in rec["ingredients"]:
                item = ing["item"]
                qty = ing.get("qty", 1.0)
                unit = ing.get("unit", "")
                key = (item.lower(), unit)
                totals[key] += qty
    rows = [{"item": item.title(), "quantity": round(qty, 2), "unit": unit}
            for (item, unit), qty in sorted(totals.items())]
    return pd.DataFrame(rows)


def plan_to_dataframe(plan: Dict[int, List[Recipe]], meals_per_day: int) -> pd.DataFrame:
    rows = []
    slot_names = get_day_slots(meals_per_day)
    for day, meals in plan.items():
        for i, r in enumerate(meals, start=1):
            label = slot_names[i - 1] if i - 1 < len(slot_names) else f"Meal {i}"
            if r is None:
                rows.append({"day": day, "meal": label, "recipe": "(empty)", "calories": 0,
                             "protein_g": 0, "carbs_g": 0, "fat_g": 0})
            else:
                rows.append({
                    "day": day,
                    "meal": label,
                    "recipe": r["name"],
                    "calories": r["calories"],
                    "protein_g": r["macros"]["protein_g"],
                    "carbs_g": r["macros"]["carbs_g"],
                    "fat_g": r["macros"]["fat_g"],
                })
    return pd.DataFrame(rows)


def generate_pdf(plan_df: pd.DataFrame, shop_df: pd.DataFrame, title: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    def header(text):
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.darkgreen)
        c.drawString(1 * inch, height - 1 * inch, text)
        c.setFillColor(colors.black)

    def table(df: pd.DataFrame, start_y: float, max_rows: int = 28, col_sizes: List[float] | None = None):
        x = 0.8 * inch
        y = start_y
        c.setFont("Helvetica", 9)
        cols = list(df.columns)
        if not col_sizes:
            total_w = width - 1.6 * inch
            col_sizes = [total_w / len(cols)] * len(cols)
        # header
        c.setFont("Helvetica-Bold", 9)
        for i, col in enumerate(cols):
            c.drawString(x + sum(col_sizes[:i]) + 2, y, str(col))
        c.setFont("Helvetica", 9)
        y -= 14
        rows_drawn = 0
        for _, row in df.iterrows():
            if rows_drawn >= max_rows or y < 1 * inch:
                c.showPage()
                y = height - 1 * inch
                # redraw header
                c.setFont("Helvetica-Bold", 9)
                for i, col in enumerate(cols):
                    c.drawString(x + sum(col_sizes[:i]) + 2, y, str(col))
                c.setFont("Helvetica", 9)
                y -= 14
                rows_drawn = 0
            for i, col in enumerate(cols):
                c.drawString(x + sum(col_sizes[:i]) + 2, y, str(row[col]))
            y -= 12
            rows_drawn += 1
        return y

    # Page 1: Plan
    header(title)
    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, height - 1.3 * inch, "Weekly Meal Plan")
    plan_small = plan_df.copy()
    plan_small["recipe"] = plan_small["recipe"].str.slice(0, 30)
    y = height - 1.55 * inch
    table(plan_small, y)

    c.showPage()
    # Page 2: Shopping list
    header("Shopping List")
    y2 = height - 1.3 * inch
    shop_small = shop_df.rename(columns={"item": "Item", "quantity": "Qty", "unit": "Unit"})
    table(shop_small, y2)
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# --- Action: Generate Plan ---
excl = normalize_tokens(exclusions) + normalize_tokens(allergies)
filtered = [r for r in RECIPE_DB if recipe_matches(r, diet_flags, normalize_tokens(allergies), normalize_tokens(exclusions), cuisines)]

if not filtered:
    st.warning("No recipes match your filters. Try removing some restrictions 🤔")
else:
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.subheader(f"Your {days}-day plan")
        plan = pick_meals(filtered, meals_per_day, days, st.session_state.calorie_target if st.session_state.is_premium else None)
        # Keep filtered recipes & plan in session for swaps/removals
        if "filtered_recipes" not in st.session_state:
            st.session_state.filtered_recipes = filtered
        if "slots" not in st.session_state:
            st.session_state.slots = get_day_slots(meals_per_day)
        
        # Initialize plan only if missing or inputs changed materially
        regen_needed = (
            "plan" not in st.session_state
            or st.session_state.get("meals_per_day_prev") != meals_per_day
            or st.session_state.get("days_prev") != days
        )
        if regen_needed:
            st.session_state.plan = pick_meals(
                filtered, meals_per_day, days,
                st.session_state.calorie_target if st.session_state.is_premium else None
            )
            st.session_state.meals_per_day_prev = meals_per_day
            st.session_state.days_prev = days
        
        plan = st.session_state.plan
        df_plan = plan_to_dataframe(plan, meals_per_day)
        st.dataframe(df_plan, use_container_width=True, hide_index=True)
        # --- Recipe viewing section ---
        st.subheader("View Recipes")
        for _, row in df_plan.iterrows():
            recipe_name = row["recipe"]
            recipe_data = next((r for r in RECIPE_DB if r["name"] == recipe_name), None)
            if recipe_data:
                with st.expander(recipe_name):
                    st.write("**Ingredients:**")
                    for ing in recipe_data["ingredients"]:
                        st.write(f"- {ing.get('qty', '')} {ing.get('unit', '')} {ing['item']}")
                    st.write("**Steps:**")
                    for i, step in enumerate(recipe_data.get("steps", []), start=1):
                        st.write(f"{i}. {step}")
        
                if st.session_state.is_premium:
                    # Per-day summaries
                    day_summary = df_plan.groupby("day")[["calories", "protein_g", "carbs_g", "fat_g"]].sum().reset_index()
                    st.markdown("**Daily totals**")
                    st.dataframe(day_summary, use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Shopping list")
                df_shop = consolidate_shopping_list(plan)
                st.dataframe(df_shop, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("---")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_plan = df_plan.to_csv(index=False).encode()
        st.download_button("Download Plan (CSV)", data=csv_plan, file_name="mealplan.csv", mime="text/csv")
    with dl2:
        csv_shop = df_shop.to_csv(index=False).encode()
        st.download_button("Download Shopping List (CSV)", data=csv_shop, file_name="shopping_list.csv", mime="text/csv")
    with dl3:
        if st.session_state.is_premium:
            pdf_bytes = generate_pdf(df_plan, df_shop, f"{APP_NAME} — Personalized Plan")
            st.download_button("Download PDF (Premium)", data=pdf_bytes, file_name="mealplan.pdf", mime="application/pdf")
        else:
            st.button("Download PDF (Premium)", disabled=True, help="Upgrade to unlock PDF export")

st.caption("Built with ❤️ in Streamlit. Not medical advice. Consult a professional for clinical nutrition.")

st.markdown("### Edit your plan (Premium)")
if not st.session_state.is_premium:
    st.info("Swapping and removing meals is a Premium feature. Upgrade to unlock!")
else:
    import random

    # Build course buckets once for fast swaps
    filtered = st.session_state.filtered_recipes
    buckets: Dict[str, List[Recipe]] = {"breakfast": [], "lunch": [], "dinner": [], "any": []}
    for r in filtered:
        buckets.setdefault(r.get("course", "any"), buckets["any"]).append(r)

    def slot_key_from_label(label: str) -> str:
        return {"Breakfast": "breakfast", "Lunch": "lunch", "Dinner": "dinner", "Snack": "any"}.get(label, "any")

    def candidates_for(slot_key: str) -> List[Recipe]:
        # prefer exact course, then 'any'
        seen = set()
        out = []
        for r in buckets.get(slot_key, []):
            if r["name"] not in seen:
                out.append(r); seen.add(r["name"])
        for r in buckets.get("any", []):
            if r["name"] not in seen:
                out.append(r); seen.add(r["name"])
        return out

    slots = st.session_state.slots

    # UI per day
    for day, meals in plan.items():
        with st.expander(f"Day {day}"):
            used_today = {m["name"] for m in meals if m}
            for idx, current in enumerate(meals):
                label = slots[idx] if idx < len(slots) else f"Meal {idx+1}"
                col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
                with col1:
                    st.write(f"**{label}** — {current['name'] if current else '(empty)'}")
                with col2:
                    if st.button(f"Swap", key=f"swap-{day}-{idx}"):
                        slot_key = slot_key_from_label(label)
                        cands = [r for r in candidates_for(slot_key)
                                 if r and r["name"] not in used_today and (not current or r["name"] != current["name"])]
                        random.shuffle(cands)
                        if cands:
                            choice = cands[0]
                            st.session_state.plan[day][idx] = choice
                            used_today.add(choice["name"])
                            st.rerun()
                        else:
                            st.warning("No alternative found for this slot.")
                with col3:
                    if st.button("Remove", key=f"remove-{day}-{idx}"):
                        st.session_state.plan[day][idx] = None
                        st.rerun()

