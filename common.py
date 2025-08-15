# common.py
import os, time, json, random, re
from typing import List, Dict, Tuple, Any
import streamlit as st
import pandas as pd
import requests

from recipe_db import RECIPE_DB, Recipe  # keep using Recipe from recipe_db

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
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # Many OpenRouter models support this; if they don't, it's ignored.
        "response_format": {"type": "json_object"},
    }
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
        if d not in recipe["diet_tags"]:
            return False
    ing_tokens = [i["item"].lower() for i in recipe["ingredients"]]
    for bad in allergies + exclusions:
        if any(bad in tok for tok in ing_tokens):
            return False
    if cuisines and recipe.get("cuisine") not in cuisines:
        return False
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
# === AI: generate brand-new recipes with ingredients & steps ===
def generate_ai_menu_with_recipes(
    days: int,
    meals_per_day: int,
    diets: list[str] | None = None,
    allergies: list[str] | None = None,
    exclusions: list[str] | None = None,
    cuisines: list[str] | None = None,
    calorie_target: int | None = None,
    model: str = OPENROUTER_MODEL,
) -> dict[int, list[dict]]:
    """
    Ask the model to CREATE recipes + a weekly plan and return a dict:
      { 1: [recipe_like, ...], 2: [...], ... }
    If the model doesn't return usable JSON, this function shows a clear error
    and returns {} (so the UI can tell you it failed).
    """
    slots = get_day_slots(meals_per_day)
    constraints = {
        "days": days,
        "slots": slots,  # e.g. ["Breakfast","Lunch","Dinner"]
        "diets": diets or [],
        "allergies": allergies or [],
        "exclusions": exclusions or [],
        "cuisines": cuisines or [],
        "daily_calorie_target": calorie_target,
    }

    # What we want back
    schema = {
      "type": "object",
      "properties": {
        "plan": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "day": {"type": "integer"},
              "meals": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "slot": {"type": "string"},
                    "recipe": {
                      "type": "object",
                      "properties": {
                        "name": {"type": "string"},
                        "course": {"type": "string"},
                        "cuisine": {"type": "string"},
                        "calories": {"type": "integer"},
                        "macros": {
                          "type": "object",
                          "properties": {
                            "protein_g": {"type": "integer"},
                            "carbs_g": {"type": "integer"},
                            "fat_g": {"type": "integer"}
                          }
                        },
                        "ingredients": {
                          "type": "array",
                          "items": {
                            "type": "object",
                            "properties": {
                              "item": {"type": "string"},
                              "qty": {"type": ["number","string"]},
                              "unit": {"type": "string"}
                            },
                            "required": ["item"]
                          }
                        },
                        "steps": {"type": "array", "items": {"type": "string"}}
                      },
                      "required": ["name","ingredients","steps"]
                    }
                  },
                  "required": ["slot","recipe"]
                }
              }
            },
            "required": ["day","meals"]
          }
        }
      },
      "required": ["plan"]
    }

    system_msg = (
        "You are a meal-planning chef. Generate complete, practical recipes "
        "that use widely available ingredients. Use simple units (cup, tbsp, tsp, g). "
        "Respect dietary/allergy constraints. Match the requested slots "
        "(breakfast/lunch/dinner). Aim near the daily calorie target if provided.\n\n"
        "RETURN ONLY RAW JSON. Do NOT wrap in code fences. Do NOT add commentary."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                "Schema (strict):\n"
                f"{json.dumps(schema)}\n\n"
                "Constraints:\n"
                f"{json.dumps(constraints)}\n\n"
                "Again, return ONLY the JSON object that matches the schema above."
            ),
        },
    ]

    # ---- Call model
    try:
        resp = call_openrouter(messages, model=model, max_tokens=2200)
        raw = (resp.get("choices", [{}])[0]
                   .get("message", {})
                   .get("content", "") or "")
    except Exception as e:
        st.error(f"AI request failed: {e}")
        return {}

    # ---- Extract JSON from possible markdown / extra text
    import re as _re
    
    def _extract_json(text: str) -> str:
        t = text.strip()
        if "```" in t:
            m = _re.search(r"```json\s*(.+?)```", t, flags=_re.S | _re.I)
            if m: 
                return m.group(1).strip()
            m = _re.search(r"```(\s*.+?)```", t, flags=_re.S)
            if m:
                return m.group(1).strip()
        if "{" in t and "}" in t:
            start = t.find("{")
            end = t.rfind("}")
            return t[start:end+1]
        return t
    
    def _clean_json(text: str) -> str:
        # normalize quotes / spaces
        t = (text
             .replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
             .replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
             .replace("\xa0", " ")                           # non‑breaking space
        )
        # remove trailing commas before ] or }
        t = _re.sub(r",\s*(\])", r"\1", t, flags=_re.S)
        t = _re.sub(r",\s*(\})", r"\1", t, flags=_re.S)
        return t.strip()
    
    cleaned = _clean_json(_extract_json(raw))
    
    # ---- Parse and normalize (with a fallback attempt)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # One more gentle pass: collapse repeated commas/newlines, then try again
        cleaned2 = _re.sub(r",\s*,", ",", cleaned)
        try:
            parsed = json.loads(cleaned2)
        except Exception as e:
            st.error("AI returned non‑JSON or invalid JSON. Enable **Generate new recipes** again to retry.")
            with st.expander("Show AI raw output"):
                st.code(raw[:6000])
            with st.expander("Show cleaned JSON we tried to parse"):
                st.code(cleaned[:6000], language="json")
            return {}

    plan_dict: dict[int, list[dict]] = {}
    try:
        for block in parsed.get("plan", []):
            day = int(block.get("day", 0))
            meals_out: list[dict] = []
            for m in block.get("meals", []):
                r = m.get("recipe", {}) or {}
                meals_out.append({
                    "name": r.get("name", "Recipe"),
                    "course": (r.get("course") or m.get("slot") or "any").lower(),
                    "cuisine": r.get("cuisine", ""),
                    "calories": int(r.get("calories") or 0),
                    "macros": {
                        "protein_g": int((r.get("macros") or {}).get("protein_g") or 0),
                        "carbs_g":   int((r.get("macros") or {}).get("carbs_g")   or 0),
                        "fat_g":     int((r.get("macros") or {}).get("fat_g")     or 0),
                    },
                    "ingredients": [
                        {
                            "item": str(i.get("item","")).strip(),
                            "qty":  i.get("qty", 1),
                            "unit": str(i.get("unit","")).strip(),
                        }
                        for i in (r.get("ingredients") or [])
                        if str(i.get("item","")).strip()
                    ],
                    "steps": [str(s).strip() for s in (r.get("steps") or []) if str(s).strip()],
                })
            if day and meals_out:
                plan_dict[day] = meals_out
    except Exception as e:
        st.error(f"Could not normalize AI output: {e}")
        with st.expander("Show parsed JSON"):
            st.code(json.dumps(parsed, indent=2))
        return {}

    if not plan_dict:
        st.error("AI didn’t return any usable meals. Try again or switch to 'Pick from built‑in'.")
        with st.expander("Show parsed JSON"):
            st.code(json.dumps(parsed, indent=2))
    return plan_dict

# ===== Plan → DataFrame & Shopping list =====

from typing import Dict, List, Tuple
import pandas as _pd

def plan_to_dataframe(plan: Dict[int, List[dict]], meals_per_day: int) -> _pd.DataFrame:
    """
    Convert the plan dict {day: [recipe_like,...]} to a flat DataFrame.
    Works for both DB-picked recipes and AI-generated ones.
    """
    rows = []
    slots = get_day_slots(meals_per_day)
    for day, meals in (plan or {}).items():
        for i, r in enumerate(meals or [], start=1):
            label = slots[i-1] if i-1 < len(slots) else f"Meal {i}"
            if not r:
                rows.append({
                    "day": day, "meal": label, "recipe": "(empty)",
                    "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0
                })
                continue
            macros = r.get("macros", {}) or {}
            rows.append({
                "day": day,
                "meal": label,
                "recipe": r.get("name", ""),
                "calories": int(r.get("calories", 0) or 0),
                "protein_g": int(macros.get("protein_g", 0) or 0),
                "carbs_g":   int(macros.get("carbs_g", 0) or 0),
                "fat_g":     int(macros.get("fat_g", 0) or 0),
            })
    return _pd.DataFrame(rows)

def consolidate_shopping_list(plan: Dict[int, List[dict]]) -> _pd.DataFrame:
    """
    Aggregate ingredients across the whole plan into a shopping list.
    Robust to missing qty/unit/types.
    """
    from collections import defaultdict
    totals: Dict[Tuple[str, str], float] = defaultdict(float)

    for meals in (plan or {}).values():
        for rec in meals or []:
            if not rec:
                continue
            for ing in rec.get("ingredients", []) or []:
                item = str(ing.get("item", "")).strip()
                if not item:
                    continue
                unit = str(ing.get("unit", "")).strip()
                # try to parse numeric qty; default to 1.0
                qty = ing.get("qty", 1.0)
                try:
                    qty = float(qty)
                except Exception:
                    qty = 1.0
                key = (item.lower(), unit)
                totals[key] += qty

    rows = [{"item": item.title(), "quantity": round(qty, 2), "unit": unit}
            for (item, unit), qty in sorted(totals.items())]
    return _pd.DataFrame(rows)

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
      - need_df = items to buy
      - have_df = pantry matches
    If annotate_at_bottom=True, pantry rows also remain in need_df (so you can label '(have)' later).
    """
    if df_shop is None or df_shop.empty:
        return df_shop, pd.DataFrame(columns=df_shop.columns if df_shop is not None else ["item","quantity","unit"])

    norm_map = {idx: _normalize_item_name(str(row["item"])) for idx, row in df_shop.iterrows()}
    pantry_norm = [_normalize_item_name(p) for p in pantry_items]

    need_rows, have_rows = [], []
    for idx, row in df_shop.iterrows():
        norm_item = norm_map[idx]
        matched = any(p and (p in norm_item or norm_item in p) for p in pantry_norm)
        if matched:
            have_rows.append(row)
            if annotate_at_bottom:
                need_rows.append(row)
        else:
            need_rows.append(row)

    need_df = pd.DataFrame(need_rows).reset_index(drop=True)
    have_df = pd.DataFrame(have_rows).reset_index(drop=True)
    return need_df, have_df
