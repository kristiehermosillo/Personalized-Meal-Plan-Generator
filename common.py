# common.py
import os, time, json, random
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
import requests

from recipe_db import RECIPE_DB, Recipe

APP_NAME = "MealPlan Genie"
FREE_DAYS = 3
PREMIUM_DAYS = 7

# Backend URL (Stripe etc.)
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Try to wake Render backend (no-op if not deployed)
try:
    requests.get(f"{DEFAULT_BACKEND_URL}/health", timeout=5)
except Exception:
    pass

# ---- AI (OpenRouter) optional ----
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"

def call_openrouter(messages: list[dict], model: str = OPENROUTER_MODEL, max_tokens: int = 1200) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.4, "max_tokens": max_tokens}
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# ---- helpers ----
def get_day_slots(meals_per_day: int) -> list[str]:
    if meals_per_day <= 2: return ["Breakfast", "Dinner"]
    if meals_per_day == 3: return ["Breakfast", "Lunch", "Dinner"]
    return ["Breakfast", "Lunch", "Dinner", "Snack"]

def normalize_tokens(s: str) -> List[str]:
    if not s: return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def recipe_matches(recipe: Recipe, diets: List[str], allergies: List[str], exclusions: List[str], cuisines: List[str]) -> bool:
    for d in diets:
        if d not in recipe["diet_tags"]: return False
    ing_tokens = [i["item"].lower() for i in recipe["ingredients"]]
    for bad in allergies + exclusions:
        if any(bad in tok for tok in ing_tokens): return False
    if cuisines and recipe.get("cuisine") not in cuisines: return False
    return True

def pick_meals(filtered: List[Recipe], meals_per_day: int, days: int, cal_target: int | None) -> Dict[int, List[Recipe]]:
    random.seed(42)
    buckets: Dict[str, List[Recipe]] = {"breakfast": [], "lunch": [], "dinner": [], "any": []}
    for r in filtered:
        buckets.setdefault(r.get("course", "any"), buckets["any"]).append(r)

    slot_keys = []
    for name in get_day_slots(meals_per_day):
        slot_keys.append({"Breakfast":"breakfast","Lunch":"lunch","Dinner":"dinner"}.get(name,"any"))

    plan: Dict[int, List[Recipe]] = {}
    for d in range(1, days+1):
        used_today = set()
        meals_today: List[Recipe] = []
        current_cals = 0
        for i, slot_key in enumerate(slot_keys, start=1):
            cands = [r for r in (buckets.get(slot_key, []) + buckets.get("any", [])) if r["name"] not in used_today]
            if not cands: cands = [r for r in filtered if r["name"] not in used_today]
            if not cands: continue
            if cal_target:
                remaining_slots = len(slot_keys)-(i-1)
                desired = max(180, int((cal_target-current_cals)/max(1, remaining_slots)))
                cands.sort(key=lambda r: abs(r["calories"]-desired))
                choice = cands[0]
            else:
                random.shuffle(cands); choice = cands[0]
            meals_today.append(choice)
            used_today.add(choice["name"])
            current_cals += choice["calories"]
        plan[d] = meals_today
    return plan

def pick_meals_ai(filtered: List[Recipe], meals_per_day: int, days: int, cal_target: int | None) -> Dict[int, List[Recipe]]:
    # Build compact catalog
    catalog = [{"name": r["name"], "course": r.get("course","any"), "cal": r["calories"]} for r in filtered]
    slots = get_day_slots(meals_per_day)
    schema = {"type":"object","properties":{"plan":{"type":"array","items":{
        "type":"object","properties":{"day":{"type":"integer"},"meals":{"type":"array","items":{
            "type":"object","properties":{"slot":{"type":"string"},"recipe_name":{"type":"string"}},
            "required":["slot","recipe_name"]}}},"required":["day","meals"]}}},"required":["plan"]}
    sys = ("Select a weekly plan strictly from the provided recipes. "
           "Respect course slots. Do not invent names. If target provided, aim near it.")
    user_payload = {"catalog":catalog,"days":days,"slots_per_day":slots,"daily_calorie_target":cal_target,"schema":schema}
    messages = [
        {"role":"system","content":sys},
        {"role":"user","content":f"Return ONLY JSON matching this schema:\n{json.dumps(schema)}\n\nInput:\n{json.dumps(user_payload)}"}
    ]
    try:
        data = call_openrouter(messages)
        text = data["choices"][0]["message"]["content"].strip().strip("`")
        if text.lower().startswith("json"): text = text[4:].strip()
        parsed = json.loads(text)
        name_map = {r["name"]: r for r in filtered}
        plan: Dict[int, List[Recipe]] = {}
        for block in parsed.get("plan", []):
            meals = [name_map.get(m["recipe_name"]) for m in block.get("meals", [])]
            plan[int(block["day"])] = [r for r in meals if r]
        return plan or pick_meals(filtered, meals_per_day, days, cal_target)
    except Exception as e:
        st.warning(f"AI planner failed; using default. {e}")
        return pick_meals(filtered, meals_per_day, days, cal_target)

def plan_to_dataframe(plan: Dict[int, List[Recipe]], meals_per_day: int) -> pd.DataFrame:
    rows = []
    slots = get_day_slots(meals_per_day)
    for day, meals in plan.items():
        for i, r in enumerate(meals, start=1):
            label = slots[i-1] if i-1 < len(slots) else f"Meal {i}"
            if r is None:
                rows.append({"day":day,"meal":label,"recipe":"(empty)","calories":0,"protein_g":0,"carbs_g":0,"fat_g":0})
            else:
                rows.append({"day":day,"meal":label,"recipe":r["name"],"calories":r["calories"],
                             "protein_g":r["macros"]["protein_g"],"carbs_g":r["macros"]["carbs_g"],"fat_g":r["macros"]["fat_g"]})
    return pd.DataFrame(rows)

def consolidate_shopping_list(plan: Dict[int, List[Recipe]]) -> pd.DataFrame:
    from collections import defaultdict
    totals: Dict[Tuple[str,str], float] = defaultdict(float)
    for meals in plan.values():
        for rec in meals:
            if not rec: continue
            for ing in rec["ingredients"]:
                key = (ing["item"].lower(), ing.get("unit",""))
                totals[key] += float(ing.get("qty", 1.0))
    rows = [{"item": item.title(),"quantity": round(qty,2),"unit": unit} for (item,unit),qty in sorted(totals.items())]
    return pd.DataFrame(rows)

def ensure_plan_exists():
    """Guard for subpages: if no plan in session, ask user to go to Home."""
    if "plan" not in st.session_state:
        st.warning("No plan yet. Go to **Home** and generate a plan first.")
        st.stop()
        
# =======================
# Shopping/plan utilities
# =======================
from typing import List, Dict, Tuple
import pandas as pd
import re

# NOTE: 'Recipe' typing may already be defined in your file.
# If not, uncomment this:
Recipe = Dict[str, any]

def plan_to_dataframe(plan: Dict[int, List[dict]], meals_per_day: int) -> pd.DataFrame:
    """Convert plan dict -> rows for display."""
    rows = []
    # Reuse your slot helper if it's in common.py, else provide fallback:
    try:
        slot_names = get_day_slots(meals_per_day)  # existing helper in your repo
    except NameError:
        # Fallback if not imported above
        def _get_day_slots(n: int) -> list[str]:
            return ["Breakfast", "Lunch", "Dinner"] if n == 3 else (
                ["Breakfast", "Dinner"] if n <= 2 else
                ["Breakfast", "Lunch", "Dinner", "Snack"]
            )
        slot_names = _get_day_slots(meals_per_day)

    for day, meals in plan.items():
        for i, r in enumerate(meals, start=1):
            label = slot_names[i - 1] if i - 1 < len(slot_names) else f"Meal {i}"
            if not r:
                rows.append({"day": day, "meal": label, "recipe": "(empty)",
                             "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0})
            else:
                rows.append({
                    "day": day,
                    "meal": label,
                    "recipe": r["name"],
                    "calories": r.get("calories", 0),
                    "protein_g": r.get("macros", {}).get("protein_g", 0),
                    "carbs_g": r.get("macros", {}).get("carbs_g", 0),
                    "fat_g": r.get("macros", {}).get("fat_g", 0),
                })
    return pd.DataFrame(rows)


def consolidate_shopping_list(plan: Dict[int, List[dict]]) -> pd.DataFrame:
    """Aggregate ingredients across the whole plan."""
    from collections import defaultdict
    totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for meals in plan.values():
        for rec in meals:
            if not rec:
                continue
            for ing in rec.get("ingredients", []):
                item = str(ing.get("item", "")).strip()
                qty = ing.get("qty", 1.0)
                try:
                    qty = float(qty)
                except Exception:
                    qty = 1.0
                unit = str(ing.get("unit", "")).strip()
                key = (item.lower(), unit)
                totals[key] += qty
    rows = [{"item": item.title(), "quantity": round(qty, 2), "unit": unit}
            for (item, unit), qty in sorted(totals.items())]
    return pd.DataFrame(rows)


# -------------------
# Pantry helpers
# -------------------
def _normalize_item_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)     # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("es") and len(s) > 3:
        s = s[:-2]
    elif s.endswith("s") and len(s) > 3:
        s = s[:-1]
    return s

def parse_pantry_text(text: str) -> list[str]:
    """Accept comma- or newline-separated pantry items."""
    if not text:
        return []
    raw = re.split(r"[,\n]", text)
    return [i.strip() for i in raw if i.strip()]

def split_shopping_by_pantry(df_shop: pd.DataFrame, pantry_items: list[str], annotate_at_bottom: bool = False):
    """
    Returns (need_df, have_df).
      - 'need_df' = items to buy
      - 'have_df' = pantry matches
    If annotate_at_bottom=True, pantry rows stay in need_df too (we can label them '(have)').
    """
    if df_shop is None or df_shop.empty:
        return df_shop, pd.DataFrame(columns=df_shop.columns if df_shop is not None else ["item","quantity","unit"])

    # Build normalized lookup
    norm_map = {idx: _normalize_item_name(str(row["item"])) for idx, row in df_shop.iterrows()}
    pantry_norm = [_normalize_item_name(p) for p in pantry_items]

    need_rows, have_rows = [], []
    for idx, row in df_shop.iterrows():
        norm_item = norm_map[idx]
        matched = any(p and (p in norm_item or norm_item in p) for p in pantry_norm)
        if matched:
            have_rows.append(row)
            if annotate_at_bottom:
                need_rows.append(row)  # keep it in the main list too (we'll annotate later)
        else:
            need_rows.append(row)

    need_df = pd.DataFrame(need_rows).reset_index(drop=True)
    have_df = pd.DataFrame(have_rows).reset_index(drop=True)
    return need_df, have_df








