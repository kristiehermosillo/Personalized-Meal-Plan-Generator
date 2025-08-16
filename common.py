# common.py
import os, time, json, random, re
from typing import List, Dict, Tuple, Any
import streamlit as st
import pandas as pd
import requests

from recipe_db import RECIPE_DB, Recipe

APP_NAME = "MealPlan Genie"
FREE_DAYS = 3
PREMIUM_DAYS = 7

# Backend URL
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Try to wake backend
try:
    requests.get(f"{DEFAULT_BACKEND_URL}/health", timeout=5)
except Exception:
    pass

# AI settings
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"

def call_openrouter(messages, model=OPENROUTER_MODEL, max_tokens=1200, temperature=0):
    """Minimal OpenRouter client for Streamlit Cloud."""
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in env or secrets")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature or 0),
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

# JSON helpers used by generate_ai_menu_with_recipes
def _extract_json(text: str) -> str:
    """Extract the first full JSON object or array, aware of quotes and escapes."""
    t = (text or "").strip()

    # strip code fences first
    m = re.search(r"```(?:json)?\s*(.+?)```", t, flags=re.S | re.I)
    if m:
        t = m.group(1).strip()

    # find first { or [
    start = None
    for i, ch in enumerate(t):
        if ch in "{[":
            start = i
            break
    if start is None:
        return t

    # walk and match until the top level closes
    stack = []
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if not stack:
                    return t[start:i+1]
                want = stack.pop()
                if ch != want:
                    # mismatch, return what we have
                    return t[start:i+1]
                if not stack:
                    return t[start:i+1]
    return t[start:]

def _clean_json(s: str) -> str:
    """Lenient cleanup: normalize quotes/literals, remove stray tokens, fix commas."""
    import re as _re
    if not s:
        return s

    # normalize smart quotes → standard JSON quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    # convert Python-ish literals to JSON
    s = _re.sub(r"\bNone\b", "null", s)
    s = _re.sub(r"\bTrue\b", "true", s)
    s = _re.sub(r"\bFalse\b", "false", s)

    # sometimes models leak a list-letter like ", b. {" between items — strip it
    s = _re.sub(r",\s*[A-Za-z]\.\s*(\{)", r", \1", s)

    # remove trailing commas before ] or }
    s = _re.sub(r",\s*(\]|\})", r"\1", s)
    # collapse double commas
    s = _re.sub(r",\s*,", ",", s)
    # remove comma right after [ or {
    s = _re.sub(r"(\[|\{)\s*,\s*", r"\1", s)

    return s.strip()




def _safe_json_load(cleaned: str, *, day_idx: int | None = None, raw: str = "") -> dict:
    """Try multiple parses; raise ValueError with clear context if it still fails."""
    import json as _json
    import re as _re

    err1 = None
    # First attempt
    try:
        parsed = _json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"plan": parsed}
    except Exception as e:
        err1 = e

    # Second attempt: tighten commas again
    cleaned2 = _re.sub(r",\s*,", ",", cleaned)
    try:
        parsed = _json.loads(cleaned2)
        return parsed if isinstance(parsed, dict) else {"plan": parsed}
    except Exception as e2:
        # Last chance: if the model returned an array, wrap it
        if cleaned2.lstrip().startswith("[") and cleaned2.rstrip().endswith("]"):
            try:
                arr = _json.loads(cleaned2)
                return {"plan": arr}
            except Exception as e3:
                err_final = f"{err1} then {e2} then {e3}"
        else:
            err_final = f"{err1} then {e2}"

    ctx = f" (day {day_idx})" if day_idx else ""
    snippet = (raw or cleaned)[:6000]
    raise ValueError(f"Invalid JSON{ctx}: {err_final}\nSnippet:\n{snippet}")


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

def _find_alternative_recipe(
    *, 
    target_slot: str, 
    diets: list[str], 
    allergies: list[str], 
    exclusions: list[str], 
    cuisines: list[str], 
    seen_names: set[str]
) -> dict | None:
    """
    Fallback from your built-in RECIPE_DB:
      - respects filters
      - matches course (slot) if possible
      - not already used this week
    Returns a 'recipe-like' dict compatible with plan structure, or None.
    """
    slot_key = target_slot.lower()
    pool = [
        r for r in RECIPE_DB 
        if recipe_matches(r, diets, allergies, exclusions, cuisines)
        and r.get("name") not in seen_names
        and (r.get("course", "any").lower() in (slot_key, "any"))
    ]
    if not pool:
        return None
    # Convert a RECIPE_DB recipe into the “recipe-like” structure the app expects
    r = random.choice(pool)
    return {
        "name": r["name"],
        "course": r.get("course", "any"),
        "cuisine": r.get("cuisine", ""),
        "calories": int(r.get("calories", 0) or 0),
        "macros": {
            "protein_g": int(r.get("macros", {}).get("protein_g", 0) or 0),
            "carbs_g":   int(r.get("macros", {}).get("carbs_g", 0) or 0),
            "fat_g":     int(r.get("macros", {}).get("fat_g", 0) or 0),
        },
        "ingredients": [
            {"item": ing.get("item",""), "qty": ing.get("qty", 1), "unit": ing.get("unit","")}
            for ing in r.get("ingredients", [])
        ],
        "steps": [s for s in r.get("steps", [])],
    }

# === AI: generate brand-new recipes with ingredients & steps ===
from typing import Callable, Optional

def generate_ai_menu_with_recipes(
    days: int,
    meals_per_day: int,
    diets: list[str] | None = None,
    allergies: list[str] | None = None,
    exclusions: list[str] | None = None,
    cuisines: list[str] | None = None,
    calorie_target: int | None = None,
    model: str = OPENROUTER_MODEL,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict[int, list[dict]]:
    """Create recipes plus a weekly plan and return {day: [recipe_like,...]}.
    Sends progress heartbeats while waiting on the model so the UI can animate.
    """

    def _primary_protein(ingredients: list[dict]) -> str:
        txt = " ".join(str(i.get("item", "")).lower() for i in (ingredients or []))
        for k in ["chicken","turkey","beef","pork","salmon","tuna","shrimp",
                  "tofu","tempeh","egg","eggs","chickpea","chickpeas","lentil",
                  "lentils","beans"]:
            if k in txt:
                return k
        return ""

    import threading as _threading
    import time as _time

    def _start_ticker(step_ref: dict, total: int, note_fn=lambda: "thinking…"):
        """Background heartbeat that bumps the timestamp every ~1s."""
        stop = {"flag": False}
        def _tick():
            while not stop["flag"]:
                if progress_cb:
                    progress_cb(step_ref["value"], total, note_fn())
                _time.sleep(1.0)
        t = _threading.Thread(target=_tick, daemon=True)
        t.start()
        return stop

    slots = get_day_slots(meals_per_day)
    base_constraints = {
        "days": days,
        "slots": slots,
        "diets": diets or [],
        "allergies": allergies or [],
        "exclusions": exclusions or [],
        "cuisines": cuisines or [],
        "daily_calorie_target": calorie_target,
    }

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
        "You are a meal planning chef. Generate complete, practical recipes with common ingredients. "
        "Respect diets and allergies and the requested slots. Aim near the daily calorie target. "
        "Return JSON only. No markdown. No prose. Use double quotes everywhere. "
        'All units must be strings like "g", "tbsp", "tsp", "cup". '
        "Output ONLY strict JSON. No code fences or commentary."
    )

    plan_dict: dict[int, list[dict]] = {}
    seen_recipe_names: set[str] = set()
    seen_primary_proteins: list[str] = []

    def _normalize_day(parsed_json: dict) -> list[dict]:
        blocks = parsed_json.get("plan", [])
        if not isinstance(blocks, list) or not blocks:
            return []
        block = blocks[0]
        meals_out: list[dict] = []
        for m in block.get("meals", []):
            r = m.get("recipe", {}) or {}
            meals_out.append({
                "name": (r.get("name") or "Recipe").strip(),
                "course": (r.get("course") or m.get("slot") or "any").lower(),
                "cuisine": r.get("cuisine", ""),
                "calories": int(r.get("calories") or 0),
                "servings": int(r.get("servings") or 1),
                "macros": {
                    "protein_g": int((r.get("macros") or {}).get("protein_g") or 0),
                    "carbs_g":   int((r.get("macros") or {}).get("carbs_g") or 0),
                    "fat_g":     int((r.get("macros") or {}).get("fat_g") or 0),
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
        return meals_out

    if progress_cb:
        progress_cb(0, days, "starting")

    for day_idx in range(1, days + 1):
        step_ref = {"value": day_idx - 1}
        if progress_cb:
            progress_cb(step_ref["value"], days, f"planning day {day_idx}…")

        day_constraints = dict(base_constraints)
        day_constraints["days"] = 1
        day_constraints["force_day"] = day_idx
        day_constraints["ban_recipes"] = sorted(seen_recipe_names)
        day_constraints["avoid_primary_proteins"] = list(dict.fromkeys(seen_primary_proteins[-2:]))

        def _ask_once(extra_hint: str = "") -> tuple[str | None, dict | None]:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":
                    "Schema (strict):\n"
                    + json.dumps(schema)
                    + "\n\nConstraints for THIS response only:\n"
                    + json.dumps(day_constraints)
                    + "\n\nVariety rules:\n"
                    + "- Never repeat an exact recipe name (see 'ban_recipes').\n"
                    + "- Rotate the main protein (see 'avoid_primary_proteins').\n"
                    + "- Vary breakfasts over the week.\n"
                    + extra_hint
                    + "\n\nReturn ONLY one day's JSON (a single item in 'plan' for 'day')."
                },
            ]

            # start heartbeat while the API is running
            ticker_stop = _start_ticker(step_ref, days, note_fn=lambda: f"planning day {day_idx}…")
            try:
                resp = call_openrouter(messages, model=model, max_tokens=1800)
                raw = (resp.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
            except Exception as e:
                if progress_cb:
                    progress_cb(step_ref["value"], days, f"error: {e!s}")
                return None, None
            finally:
                ticker_stop["flag"] = True  # stop heartbeat

            cleaned = _clean_json(_extract_json(raw))
            try:
                parsed = _safe_json_load(cleaned, day_idx=day_idx, raw=raw)
            except ValueError:
                return raw, None
            return raw, parsed

        raw, parsed = _ask_once()
        if parsed is None:
            raw, parsed = _ask_once(
                extra_hint="\n- Your previous output was invalid or duplicated names. Use different recipes and match the schema exactly."
            )

        meals_out: list[dict] = []
        if parsed:
            meals_out = _normalize_day(parsed)

        if not meals_out:
            # fallback from built-in DB
            meals_out = []
            for slot in slots:
                alt = _find_alternative_recipe(
                    target_slot=slot,
                    diets=diets or [],
                    allergies=allergies or [],
                    exclusions=exclusions or [],
                    cuisines=cuisines or [],
                    seen_names=seen_recipe_names,
                )
                if alt:
                    meals_out.append(alt)

        if meals_out:
            plan_dict[day_idx] = meals_out
            for r in meals_out:
                seen_recipe_names.add(r.get("name",""))
                pp = _primary_protein(r.get("ingredients", []))
                if pp:
                    seen_primary_proteins.append(pp)

        step_ref["value"] = day_idx
        if progress_cb:
            progress_cb(day_idx, days, f"day {day_idx} ready")

    if progress_cb:
        progress_cb(days, days, "complete")

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

def consolidate_shopping_list(plan: Dict[int, List[dict]], household_size: int = 1) -> _pd.DataFrame:
    """
    Aggregate ingredients across the whole plan into a shopping list.
    Scales quantities for the given household_size using recipe['servings'] (default 1).
    """
    from collections import defaultdict
    totals: Dict[Tuple[str, str], float] = defaultdict(float)

    hh = max(1, int(household_size or 1))

    for meals in (plan or {}).values():
        for rec in meals or []:
            if not rec:
                continue

            # servings per recipe (defaults to 1 if missing)
            try:
                recipe_servings = int(rec.get("servings") or 1)
            except Exception:
                recipe_servings = 1

            # multiplier = people / servings-per-recipe
            mult = hh / max(1, recipe_servings)

            for ing in rec.get("ingredients", []) or []:
                item = str(ing.get("item", "")).strip()
                if not item:
                    continue
                unit = str(ing.get("unit", "")).strip()

                qty = ing.get("qty", 1.0)
                try:
                    qty = float(qty)
                except Exception:
                    qty = 1.0

                qty *= mult
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
