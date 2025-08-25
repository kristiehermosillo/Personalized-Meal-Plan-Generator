# app.py — Home / Dashboard
import os, time, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor
import threading, uuid, time
from streamlit_autorefresh import st_autorefresh

# ---- flags and helpers (paste near the top, after imports) ----
HIDE_BADGE = os.getenv("HIDE_BADGE", "0").lower() in ("1", "true", "yes")

# no-op dedupe if not defined elsewhere
try:
    dedupe_plan  # type: ignore
except NameError:
    def dedupe_plan(plan, filtered_recipes):
        return plan

def _job_running():
    f = st.session_state.get("bg_future")
    return bool(f and not f.done())

# Thread-safe in-process progress store
_PROGRESS = {}
_PROGRESS_LOCK = threading.Lock()

def _set_progress(job_id: str, step: int, total: int, note: str = ""):
    with _PROGRESS_LOCK:
        _PROGRESS[job_id] = {"step": step, "total": total, "note": note, "ts": time.time()}

def _get_progress(job_id: str):
    with _PROGRESS_LOCK:
        return _PROGRESS.get(job_id)

def _clear_progress(job_id: str):
    with _PROGRESS_LOCK:
        _PROGRESS.pop(job_id, None)

@st.cache_resource
def _get_executor():
    # one executor per server process; safe across reruns
    return ThreadPoolExecutor(max_workers=2)

DEV_MODE = str(st.secrets.get("DEV_MODE", "") or os.getenv("DEV_MODE", "")).lower() in ("1", "true", "yes", "on")

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
        
def _bg_run_generation(payload: dict, filtered_recipes: list[dict], job_id: str):
    def cb(step: int, total: int, note: str = ""):
        _set_progress(job_id, step, total, note)

    if payload["use_ai"]:
        cb(0, payload["days"], "starting")  # <-- add this
        plan = generate_ai_menu_with_recipes(
            days=payload["days"],
            meals_per_day=payload["meals_per_day"],
            diets=payload["diets"],
            allergies=payload["allergies"],
            exclusions=payload["exclusions"],
            cuisines=payload["cuisines"],
            calorie_target=payload["cal_target"],
            progress_cb=cb,
        )
        cb(payload["days"], payload["days"], "complete")  # <-- and this
        return plan
    else:
        cb(0, payload["days"], "starting")
        plan = pick_meals(
            filtered_recipes,
            payload["meals_per_day"],
            payload["days"],
            payload["cal_target"],
        )
        cb(payload["days"], payload["days"], "complete")
        return plan

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
    requests.get(f"{DEFAULT_BACKEND_URL}/health", timeout=3)
except Exception:
    pass

# ---- Session bootstrap ----
for k, v in {
    "is_premium": False,
    "calorie_target": 2000,
    "last_session_id": None,
    "used_ai_prev": False,
    "plan": {},              # <- add this so plan always exists
    "filters_sig": None,
    "plan_locked": False,
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
    st.write("")  # spacer
    # Always-visible "Open shopping list" button in header
    if st.button("🛒 Open shopping list", key="btn_jump_shop_hdr", use_container_width=True):
        st.session_state["jump_to_shop"] = True
        st.session_state["_pending_nav_to_weekly"] = True   # one-shot redirect
        st.rerun()

st.divider()

# ---- Sidebar: TOP controls ----
st.sidebar.markdown("### Navigation")
NAV_OPTS = ["Today", "Weekly Overview", "Recipes"]

# One-time redirect (must run BEFORE the radio)
if st.session_state.pop("_pending_nav_to_weekly", False):
    st.session_state["nav_view"] = "Weekly Overview"

# Default once
st.session_state.setdefault("nav_view", NAV_OPTS[0])

# The radio controls the page and stores in the same key
view = st.sidebar.radio("Go to", NAV_OPTS, key="nav_view")

st.sidebar.markdown("### Plan controls")
st.session_state.plan_locked = st.sidebar.checkbox(
    "🔒 Freeze this plan",
    value=st.session_state.get("plan_locked", False),
    help="When frozen, your plan won’t change until you click Generate again."
)

st.sidebar.caption("Free: 3-day plan preview. Upgrade for 7 days + macros + PDF export.")

# --- Dev tools (hidden for users) ---
uploaded = None
if DEV_MODE:
    if st.session_state.get("plan"):
        dl_plan = json.dumps(st.session_state.plan, ensure_ascii=False, indent=0).encode()
        st.sidebar.download_button(
            "⬇️ Export plan (JSON)",
            data=dl_plan,
            file_name="mealplan.json",
            mime="application/json",
            use_container_width=True
        )
    uploaded = st.sidebar.file_uploader(
        "⬆️ Import saved plan (JSON)",
        type=["json"],
        help="Use a file exported with the button above."
    )
# ----- end dev tools -----


st.sidebar.header("Your Preferences")
# People you are cooking for
st.session_state.household_size = int(st.sidebar.number_input(
    "People you are cooking for",
    min_value=1,
    max_value=12,
    value=int(st.session_state.get("household_size", 1)),
    step=1,
    key="sidebar_household_size",            # <-- added
    help="Used to scale the shopping list"
))

diet_flags = st.sidebar.multiselect("Diet",
    ["vegetarian","vegan","gluten-free","dairy-free","pescatarian","low-carb"], default=[])
allergies  = st.sidebar.text_input("Allergies to avoid", "")
exclusions = st.sidebar.text_input("Skip ingredients", "")
meals_per_day = st.sidebar.slider("Meals per day", 2, 4, 3)
if st.session_state.is_premium:
    st.session_state.calorie_target = int(st.sidebar.number_input(
        "Daily calorie target",
        min_value=800,          # adjust if you want a wider range
        max_value=5000,
        value=int(st.session_state.get("calorie_target", 2000)),
        step=10,
        help="Type an exact number (kcal) — e.g., 1875."
    ))
else:
    st.sidebar.info("Calorie targeting available in Premium.")
cuisines = st.sidebar.multiselect("Favorite cuisines (optional)",
    ["american","mediterranean","asian","mexican","indian","middle-eastern","italian"], default=[])

days = PREMIUM_DAYS if st.session_state.is_premium else FREE_DAYS

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
import hashlib

st.subheader(f"Your {days}-day plan")

c_l, c_r = st.columns([0.6, 0.4])
with c_r:
    if st.button("🛒 Open shopping list", use_container_width=True, key="btn_jump_shop"):
        # Next run: switch page before the radio is created, and focus the list tab
        st.session_state["jump_to_shop"] = True
        st.session_state["_pending_nav_to_weekly"] = True
        st.rerun()

# Friendlier wording for normal users (no “AI” language)
if st.session_state.is_premium:
    source_choice = st.radio(
        "Recipe source",
        ["From our cookbook", "Create new recipes"],
        index=0,
        horizontal=True,
        help="“From our cookbook” uses our built-in recipes. “Create new recipes” crafts new dishes for you."
    )
    use_ai = (source_choice == "Create new recipes")
else:
    use_ai = False
    st.caption("Upgrade to create brand-new recipes each week.")

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

# ---- Always-visible Generate / Regenerate (runs in background) ----
# Show this bar when a plan already exists OR a job is running.
show_generate_bar = bool(st.session_state.get("plan")) or bool(st.session_state.get("bg_future"))

if show_generate_bar:
    busy = _job_running()
    label = "🍳 Cooking your plan…" if busy else "🍽️ Build my plan"


    if st.button(label, type="primary", use_container_width=True, key="btn_generate_top", disabled=busy):
        payload = dict(
            use_ai=bool(use_ai),
            days=days,
            meals_per_day=meals_per_day,
            diets=list(diet_flags),
            allergies=normalize_tokens(allergies),
            exclusions=normalize_tokens(exclusions),
            cuisines=list(cuisines),
            cal_target=st.session_state.calorie_target if st.session_state.is_premium else None,
            sig=sig,
        )
        job_id = uuid.uuid4().hex
        st.session_state["bg_job_id"] = job_id
        _set_progress(job_id, 0, days, "starting")
        st.session_state["bg_started_ts"] = time.time()        # <— ADD THIS
        st.session_state["spin_i"] = 0     

        # NOTE: pass job_id to the worker
        st.session_state["bg_future"] = _get_executor().submit(
            _bg_run_generation, payload, filtered, job_id
        )
        st.session_state["bg_payload"] = payload
        st.rerun()

# ---- Background watcher (runs every rerun) ----
bg_future = st.session_state.get("bg_future")
job_id = st.session_state.get("bg_job_id")

if bg_future and job_id:
    # Auto-rerun once per second while the job runs
    st_autorefresh(interval=1000, key=f"auto_refresh_{job_id}")

    # Cycle through emojis so the line visibly changes
    SPIN = ["🍳", "🔪", "🥣", "🧂", "⏲️", "🔥"]
    i = st.session_state.get("spin_i", 0)
    st.session_state["spin_i"] = (i + 1) % len(SPIN)
    emoji = SPIN[st.session_state["spin_i"]]

    # Simple “it’s working” message with elapsed time
    started = float(st.session_state.get("bg_started_ts", time.time()))
    elapsed = int(time.time() - started)
    dots = "." * (1 + (st.session_state["spin_i"] % 3))
    st.markdown(
    f"<div class='cook-banner'><span class='cook-emoji'>{emoji}</span>"
    f"<span>Cooking up your plan… <span class='cook-faint'>{elapsed}s elapsed</span></span></div>",
    unsafe_allow_html=True
)

    # When the background job finishes, collect the result and clean up
    if bg_future.done():
        try:
            result_plan = bg_future.result()
            if result_plan:
                try:
                    result_plan = dedupe_plan(result_plan, filtered)  # ok if you don't have this
                except Exception:
                    pass
                st.session_state.plan = result_plan
                st.session_state.filters_sig = st.session_state.get("bg_payload", {}).get("sig")
                st.toast("Plan ready!", icon="✅")
            else:
                st.warning("Background generation returned no plan.")
        except Exception as e:
            st.error(f"Background generation failed: {e}")
        finally:
            _clear_progress(job_id)
            st.session_state["bg_future"]  = None
            st.session_state["bg_payload"] = None
            st.session_state["bg_job_id"]  = None
            st.session_state["bg_started_ts"] = None
            st.session_state["spin_i"] = 0
            st.rerun()


# If a plan file was uploaded (Dev Mode), read it and set the current plan
if uploaded is not None:
    try:
        file_bytes = uploaded.read()
        loaded_plan = json.loads(file_bytes.decode("utf-8", errors="strict"))
        if isinstance(loaded_plan, dict):
            st.session_state.plan = loaded_plan
            # mark that the current filters match this loaded plan
            st.session_state.filters_sig = sig  # or recompute if you prefer
            st.success("Plan loaded from file.")
        else:
            st.error("Invalid plan file format (expected a JSON object).")
    except Exception as e:
        st.error(f"Could not load plan: {e}")
        
# --- Fill any missing days from our cookbook so users always get a full week
if st.session_state.get("plan"):
    missing_days = [d for d in range(1, days + 1) if not st.session_state.plan.get(d)]
    if missing_days:
        fallback = pick_meals(
            filtered, meals_per_day, len(missing_days),
            st.session_state.calorie_target if st.session_state.is_premium else None
        )
        # fallback keys are 1..N; map them onto the missing day numbers
        for i, d in enumerate(missing_days, start=1):
            st.session_state.plan[d] = fallback.get(i, [])
        st.success(f"Added {len(missing_days)} day(s) from our cookbook to complete your week.")


# Optional: if inputs changed since last generation, show a gentle nudge (no auto-generate).
existing_plan = st.session_state.get("plan")
if existing_plan and st.session_state.get("filters_sig") != sig:
    st.info("Your preferences changed. Click **Generate my meal plan** to update.")


# Use the plan from session (safe when empty)
plan = st.session_state.get("plan", {}) or {}
df_plan = plan_to_dataframe(plan, meals_per_day) if plan else pd.DataFrame()


# ---- View renderer (single-file, robust) ----
st.markdown("---")
plan = st.session_state.plan

if view == "Today":
    st.subheader("📅 Today’s Meals")

    # Empty state (no plan yet)
    if not plan or not list(plan.keys()):
        with st.container(border=True):
            st.markdown("### Let’s make your week")
            st.markdown(
                "1. Pick your **diet, allergies, and cuisines** on the left.\n"
                "2. Choose **Recipe source** (from our cookbook or create new recipes).\n"
                "3. Click **Generate my meal plan**."
            )
            clicked = st.button(
                "🍽️ Generate my meal plan",
                type="primary",
                use_container_width=True,
                key="gen_btn_empty",
            )
            if clicked:
                payload = dict(
                    use_ai=bool(use_ai),
                    days=days,
                    meals_per_day=meals_per_day,
                    diets=list(diet_flags),
                    allergies=normalize_tokens(allergies),
                    exclusions=normalize_tokens(exclusions),
                    cuisines=list(cuisines),
                    cal_target=st.session_state.calorie_target if st.session_state.is_premium else None,
                    sig=sig,
                )
                job_id = uuid.uuid4().hex
                st.session_state["bg_job_id"] = job_id
                _set_progress(job_id, 0, days, "starting")
                st.session_state["bg_started_ts"] = time.time()
                st.session_state["bg_future"] = _get_executor().submit(
                    _bg_run_generation, payload, filtered, job_id
                )
                st.session_state["bg_payload"] = payload
                st.info("Cooking your plan in the background… you can keep browsing.")
                st.rerun()
        st.stop()

    # We have a plan
    max_day = max(plan.keys())

    # --- Two-line day pills (content width, centered in columns) ---
    import datetime as _dt
    # ---- Selected day bootstrap (MUST come before the day pills) ----
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = 1
    
    # make sure it's inside 1..max_day (e.g., after a new plan is generated)
    try:
        st.session_state.selected_day = int(
            max(1, min(max_day, st.session_state.selected_day))
        )
    except Exception:
        st.session_state.selected_day = 1

    st.markdown("""
    <style>
    /* Center each pill inside its column */
    div.stButton { display: flex; justify-content: center; }
    
    /* Pill base */
    div.stButton > button{
      white-space: pre-line;        /* allow \n break: 'Day 1\\nWed 20' */
      text-align: center;
      line-height: 1.2;
      padding: 12px 16px;
      font-size: .96rem;
      border-radius: 18px;
      margin: 8px 14px;             /* BIGGER gap so pills don’t “touch” */
      border: none !important;
      color: rgba(255,255,255,.88) !important;  /* slightly softer second line */
      min-width: 130px;             /* content width, not full column */
      box-shadow: 0 0 0 1px rgba(255,255,255,.06),
                  0 4px 10px rgba(0,0,0,.25);   /* subtle separation */
    }
    
    /* Make the first line (Day X) pop */
    div.stButton > button::first-line{
      font-weight: 700;
      font-size: 1.06rem;
      color: #fff;
    }
    
    /* Unselected vs selected */
    div.stButton > button[kind="secondary"]{
      background: #343434 !important;   /* subtle filled */
    }
    div.stButton > button[kind="primary"]{
      background: #ff4b4b !important;   /* accent filled */
      box-shadow: 0 0 0 1px rgba(255,255,255,.12),
                  0 6px 14px rgba(255,75,75,.25);
    }
    
    /* Phone tweaks: wrap to 2-up, then 1-up */
    @media (max-width: 560px){
      div.stButton > button{ min-width: 46%; margin: 6px 6px; }
    }
    @media (max-width: 380px){
      div.stButton > button{ min-width: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)

    
    start = _dt.date.today()
    cols = st.columns(min(max_day, 7))
    
    for i in range(1, max_day + 1):
        with cols[(i - 1) % len(cols)]:
            date_str = (start + _dt.timedelta(days=i - 1)).strftime("%a %d")
            label = f"Day {i}\n{date_str}"
            is_selected = (i == st.session_state.selected_day)
            btn_type = "primary" if is_selected else "secondary"
    
            # NOTE: no use_container_width — lets pills be content width
            if st.button(label, key=f"daybtn_{i}", type=btn_type):
                st.session_state.selected_day = i
                st.rerun()
    
    # chosen day
    day = st.session_state.selected_day
    slots = get_day_slots(meals_per_day)
    meals = plan.get(day, [])


    
    # ---------- Meals ----------
    ICONS = ["🍳", "🥗", "🍝", "🍱"]
    for i, r in enumerate(meals, start=1):
        label = slots[i - 1] if i - 1 < len(slots) else f"Meal {i}"
        if not r:
            st.write(f"**{label}** — empty")
            continue
        icon = ICONS[(i - 1) % len(ICONS)]
        kcal = int(r.get("calories") or 0)
        serv = int(r.get("servings") or 1)
        title = f"{icon} {label} — {r.get('name','Recipe')} · {kcal} kcal · serves {serv}"
        with st.expander(title, expanded=(i == 1)):
            st.write("**Ingredients:**")
            for ing in r.get("ingredients", []):
                st.write(f"- {ing.get('qty','')} {ing.get('unit','')} {ing.get('item','')}".strip())
            st.write("**Steps:**")
            for idx, step in enumerate(r.get("steps", []), start=1):
                st.write(f"{idx}. {step}")

elif view == "Weekly Overview":
    st.subheader("🗓️ Week at a glance")

    # If the gray 'Open shopping list' button was clicked on the Home section,
    # this flag will exist just for this rerun. Use it once, then forget it.
    jump = bool(st.session_state.pop("jump_to_shop", False))

    # Build dataframes safely
    df_plan2 = plan_to_dataframe(plan, meals_per_day) if plan else pd.DataFrame()
    if df_plan2 is None or df_plan2.empty or "day" not in df_plan2.columns:
        st.info("No plan data to summarize yet. Click **Generate my meal plan** first.")
        st.stop()

    # Shopping list (scaled)
    hh_size = int(st.session_state.get("household_size", 1))
    df_shop2 = consolidate_shopping_list(plan, household_size=hh_size)

    # Pantry matching
    pantry_items = parse_pantry_text(st.session_state.get("pantry_text", ""))
    annotate = st.session_state.get("show_pantry_note", False)
    need_df, have_df = split_shopping_by_pantry(
        df_shop2, pantry_items, annotate_at_bottom=annotate
    )

    # ---------- Weekly Overview · Prep plan snapshot ----------

    import re
    from math import ceil
    
    def _norm(s: str) -> str:
        return str(s or "").strip().lower()
    
    def _short_kcal(n: int) -> str:
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)
    
    # --- 1) Prep load per day (rough) ---
    def _estimate_recipe_minutes(rec: dict) -> tuple[int, int]:
        """Return (estimated_minutes, steps_count). Use explicit '10 min' patterns if present,
        otherwise ~3 min/step with a floor to avoid 0-minute recipes."""
        steps = rec.get("steps") or []
        txt = " ".join(_norm(s) for s in steps)
        mins = 0
        for m in re.findall(r"(\d+)\s*(?:min|mins|minute|minutes)\b", txt):
            try:
                mins += int(m)
            except:
                pass
        if mins == 0:
            mins = max(10, int(len(steps) * 3))
        return mins, len(steps)
    
    day_prep = []  # list of dicts: {day, mins, steps}
    for d in range(1, days + 1):
        mins = 0
        steps = 0
        for rec in plan.get(d, []):
            if rec:
                m, s = _estimate_recipe_minutes(rec)
                mins += m
                steps += s
        day_prep.append({"day": d, "mins": mins, "steps": steps})
    
    # light/medium/heavy labelling by thirds
    sorted_by_mins = sorted(day_prep, key=lambda r: r["mins"])
    chunk = max(1, ceil(len(sorted_by_mins)/3))
    light_days  = {r["day"] for r in sorted_by_mins[:chunk]}
    heavy_days  = {r["day"] for r in sorted_by_mins[-chunk:]}
    for r in day_prep:
        if r["day"] in heavy_days:
            r["load"] = "heavy"
        elif r["day"] in light_days:
            r["load"] = "light"
        else:
            r["load"] = "medium"
    
    # --- 2) “Batch once, chop list” (common veg across multiple days) ---
    CHOP_ITEMS = [
        "onion","garlic","bell pepper","pepper","carrot","celery","broccoli",
        "cauliflower","tomato","cucumber","spinach","mushroom","cilantro",
        "parsley","ginger","green onion","scallion","kale","zucchini"
    ]
    
    def _qty_float(x):
        try:
            return float(x)
        except:
            return None
    
    chop_map = {}  # name -> {"days": set(), "totals": {unit: qty}}
    for d in range(1, days + 1):
        for rec in plan.get(d, []):
            if not rec:
                continue
            for ing in rec.get("ingredients", []) or []:
                name = _norm(ing.get("item",""))
                hit = None
                for token in CHOP_ITEMS:
                    if token in name:
                        hit = token
                        break
                if not hit:
                    continue
                entry = chop_map.setdefault(hit, {"days": set(), "totals": {}})
                entry["days"].add(d)
                q = _qty_float(ing.get("qty"))
                u = _norm(ing.get("unit",""))
                if q is not None and u:
                    entry["totals"][u] = entry["totals"].get(u, 0.0) + q
    
    batch_chop = []
    for item, data in chop_map.items():
        if len(data["days"]) >= 2:
            # prefer the most common unit if we have totals
            total_txt = ""
            if data["totals"]:
                unit, qty = max(data["totals"].items(), key=lambda kv: kv[1])
                # round nicely (avoid .0 noise)
                qty = int(qty) if abs(qty - int(qty)) < 1e-6 else round(qty, 1)
                total_txt = f" · ~{qty} {unit}"
            days_list = ", ".join(f"Day {d}" for d in sorted(data["days"]))
            batch_chop.append(f"• {item.title()} on {days_list}{total_txt}")
    
    # --- 3) “Cook once, reuse” (staples that appear on multiple days) ---
    REUSE_ITEMS = [
        "rice","quinoa","beans","lentils","chicken","tofu","pork","beef",
        "sauce","dressing","marinade","hummus","pesto","salsa","roasted"
    ]
    reuse_map = {}  # token -> sorted list of days
    for d in range(1, days + 1):
        names = [_norm((rec or {}).get("name","")) for rec in plan.get(d, [])]
        ings  = [_norm(i.get("item","")) for rec in plan.get(d, []) for i in (rec or {}).get("ingredients", []) or []]
        haystack = " | ".join(names + ings)
        for tok in REUSE_ITEMS:
            if tok in haystack:
                reuse_map.setdefault(tok, set()).add(d)
    
    reuse_suggest = []
    for tok, dset in reuse_map.items():
        ds = sorted(dset)
        if len(ds) >= 2:
            first, rest = ds[0], ds[1:]
            reuse_suggest.append(f"• Make extra **{tok}** on Day {first} → reuse on Day(s) {', '.join(map(str, rest))}")
    
    # --- 4) Make-ahead components (by recipe names/steps) ---
    COMP_TOKS = ["sauce","dressing","marinade","pesto","salsa","slaw","hummus","vinaigrette"]
    components = set()
    for d in range(1, days + 1):
        for rec in plan.get(d, []):
            if not rec:
                continue
            name = _norm(rec.get("name",""))
            step_blob = " ".join(_norm(s) for s in rec.get("steps", []) or [])
            for tok in COMP_TOKS:
                if tok in name or tok in step_blob:
                    components.add(tok)
    make_ahead = sorted(components)
    
    # --- Render ---
    st.caption(f"Showing {days} day(s) · {meals_per_day} meals/day · scaled for {hh_size} person(s)")
    
    st.markdown("### 🔧 Prep plan snapshot")
    
    c1, c2 = st.columns([0.55, 0.45], gap="large")
    
    with c1:
        st.markdown("**Prep load by day**")
        # simple visual bars relative to the heaviest day
        ref = max(1, max(r["mins"] for r in day_prep))
        for r in day_prep:
            pct = int(round(100 * r["mins"] / ref))
            if r["load"] == "light":
                label = "🟢 light"
            elif r["load"] == "heavy":
                label = "🔴 heavy"
            else:
                label = "🟡 medium"
            

    # Friendlier column names
    plan_display = df_plan2.rename(columns={
        "day": "Day",
        "meal": "Meal",
        "recipe": "Recipe",
        "calories": "Calories",
        "protein_g": "Protein (g)",
        "carbs_g": "Carbs (g)",
        "fat_g": "Fat (g)",
    })

    # Build a "Notes" column that flags allergy/skip-ingredient hits per meal
    restrict_tokens_notes = set(normalize_tokens(allergies) + normalize_tokens(exclusions))

    def _meal_note(day: int, recipe_name: str) -> str:
        if not restrict_tokens_notes:
            return ""
        for rec in plan.get(day, []):
            if rec and rec.get("name", "") == recipe_name:
                ings = " ".join(str(i.get("item", "")).lower() for i in (rec.get("ingredients") or []))
                hits = [tok for tok in restrict_tokens_notes if tok and tok in ings]
                if hits:
                    return "Contains: " + ", ".join(sorted(set(hits)))
                break
        return ""

    plan_display["Notes"] = [
        _meal_note(int(r["Day"]), str(r["Recipe"])) for _, r in plan_display.iterrows()
    ]

    # Tabs for a cleaner layout
    if jump:
        tab_shop, tab_plan, tab_totals = st.tabs(["Shopping list", "Plan table", "Daily totals"])
    else:
        tab_plan, tab_shop, tab_totals = st.tabs(["Plan table", "Shopping list", "Daily totals"])
    
            

    with tab_plan:
        st.dataframe(
            plan_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Day":         st.column_config.NumberColumn(format="%d", width="small"),
                "Meal":        st.column_config.TextColumn(width="small"),
                "Recipe":      st.column_config.TextColumn(width="large"),
                "Calories":    st.column_config.NumberColumn(format="%d", width="small"),
                "Protein (g)": st.column_config.NumberColumn(format="%d", width="small"),
                "Carbs (g)":   st.column_config.NumberColumn(format="%d", width="small"),
                "Fat (g)":     st.column_config.NumberColumn(format="%d", width="small"),
                "Notes":       st.column_config.TextColumn(width="large"),
            },
        )

    with tab_shop:
        left, right = st.columns([0.65, 0.35])

        # --- RIGHT column (controls) ---
        with right:
            new_hh = st.number_input(
                "People you’re cooking for",
                min_value=1, max_value=12, step=1, value=hh_size,
                key="shop_people",                        # <-- added
                help="Rescales quantities in the shopping list only."
            )
            if new_hh != hh_size:
                st.session_state["household_size"] = int(new_hh)
                hh_size = int(new_hh)
                df_shop2 = consolidate_shopping_list(plan, household_size=hh_size)
                need_df, have_df = split_shopping_by_pantry(
                    df_shop2, pantry_items, annotate_at_bottom=annotate
                )

            st.caption("Pantry matching")
            if pantry_items:
                st.write(f"Matched against: {', '.join(pantry_items)}")
            else:
                st.write("No pantry items entered.")

        # --- LEFT column (interactive checklist) ---
        with left:
            shop_display = need_df.copy() if annotate else need_df
            if annotate and not need_df.empty:
                norm_have = {i.lower() for i in have_df["item"].astype(str)} if not have_df.empty else set()
                shop_display["item"] = shop_display["item"].astype(str).apply(
                    lambda x: f"{x} (have)" if x.lower() in norm_have else x
                )

            if shop_display is None or shop_display.empty:
                st.info("Your shopping list is empty.")
            else:
                st.markdown("### 🛒 Shopping Checklist")

                # keep checkbox state stable across reruns
                signature = "|".join(shop_display["item"].astype(str).tolist())
                if "shop_checked" not in st.session_state or st.session_state.get("shop_keys_sig") != signature:
                    st.session_state.shop_checked = {i: False for i in range(len(shop_display))}
                    st.session_state.shop_keys_sig = signature

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Mark all", key="shop_mark_all"):
                        for i in st.session_state.shop_checked:
                            st.session_state.shop_checked[i] = True
                        st.rerun()
                with c2:
                    if st.button("Clear all", key="shop_clear_all"):
                        for i in st.session_state.shop_checked:
                            st.session_state.shop_checked[i] = False
                        st.rerun()

                import hashlib  # at top of file already imported; safe to reuse here

                # make a short, stable hash of the current list signature (you already set shop_keys_sig above)
                sig = str(st.session_state.get("shop_keys_sig", ""))
                sig_hash = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:8] if sig else "nosig"
                
                for idx, row in shop_display.reset_index(drop=True).iterrows():
                    label = f'{row["item"]} — {row["quantity"]} {row["unit"]}'.strip()
                    cb_key = f"shop_chk_{sig_hash}_{idx}"   # unique across the whole app
                    st.session_state.shop_checked[idx] = st.checkbox(
                        label,
                        value=st.session_state.shop_checked.get(idx, False),
                        key=cb_key
                    )

                # --- Copy-friendly text + Copy/Share buttons (no download) ---
                text_lines = [
                    f'- {row["item"]} — {row["quantity"]} {row["unit"]}'.strip()
                    for _, row in shop_display.iterrows()
                ]
                note_text = "Shopping List\n" + "\n".join(text_lines)
                
                st.markdown("#### 📋 Copy to your Notes app")
                
                # Use a stable, unique key so Streamlit never collides
                import hashlib as _hash
                sig = str(st.session_state.get("shop_keys_sig", ""))
                sig_hash = _hash.sha1(sig.encode("utf-8")).hexdigest()[:8] if sig else "nosig"
                
                st.text_area(
                    "Copy this list:",
                    value=note_text,
                    height=160,
                    label_visibility="collapsed",
                    key=f"shop_copy_text_{sig_hash}",
                )
                
                # Visible Copy + Share buttons (no f-string -> no brace issues)
                from streamlit.components.v1 import html as stc_html
                import json as _json
                _js_payload = _json.dumps(note_text)
                
                _html_block = """
                <div class="copy-share" style="display:flex;align-items:center;gap:10px;margin:8px 0 12px;">
                  <button id="copy_btn" style="padding:8px 12px;border-radius:8px;cursor:pointer;border:1px solid rgba(255,255,255,.15);background:#2b2b2b;color:inherit;">📋 Copy</button>
                  <button id="share_btn" style="padding:8px 12px;border-radius:8px;cursor:pointer;border:1px solid rgba(255,255,255,.15);background:#2b2b2b;color:inherit;">📤 Share…</button>
                  <span id="copy_msg" style="opacity:0;transition:opacity .2s;">Copied!</span>
                </div>
                <script>
                (function(){
                  const txt = __PAYLOAD__;
                  function showToast(){
                    const el = document.getElementById('copy_msg');
                    if(!el) return; el.style.opacity = 1; setTimeout(()=>el.style.opacity = 0, 1200);
                  }
                  function fallbackCopy(t){
                    const ta = document.createElement('textarea');
                    ta.value = t; ta.style.position='fixed'; ta.style.left='-9999px';
                    document.body.appendChild(ta); ta.focus(); ta.select();
                    try{ document.execCommand('copy'); }catch(e){}
                    document.body.removeChild(ta); showToast();
                  }
                  function copy(){
                    if(navigator.clipboard && window.isSecureContext){
                      navigator.clipboard.writeText(txt).then(showToast).catch(()=>fallbackCopy(txt));
                    } else { fallbackCopy(txt); }
                  }
                  const copyBtn = document.getElementById('copy_btn');
                  const shareBtn = document.getElementById('share_btn');
                  if(copyBtn) copyBtn.addEventListener('click', copy);
                  if(shareBtn) shareBtn.addEventListener('click', async ()=>{
                    if(navigator.share){
                      try{ await navigator.share({ text: txt, title: "Shopping List" }); }catch(e){}
                    } else { copy(); }
                  });
                })();
                </script>
                """
                stc_html(_html_block.replace("__PAYLOAD__", _js_payload), height=60)

                

                
            # Pantry matches (unchanged)
            with st.expander("Pantry items (matched)"):
                have_display = have_df.rename(
                    columns={"item": "Item", "quantity": "Quantity", "unit": "Unit"}
                )
                if have_display.empty:
                    st.write("No matches.")
                else:
                    st.dataframe(
                        have_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Item":     st.column_config.TextColumn(width="large"),
                            "Quantity": st.column_config.NumberColumn(format="%.2f", width="small"),
                            "Unit":     st.column_config.TextColumn(width="small"),
                        },
                    )

        with tab_totals:
            day_summary = (
                plan_display.groupby("Day")[["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]]
                .sum()
                .reset_index()
                .sort_values("Day")
            )
            st.dataframe(
                day_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Day":         st.column_config.NumberColumn(format="%d", width="small"),
                    "Calories":    st.column_config.NumberColumn(format="%d", width="small"),
                    "Protein (g)": st.column_config.NumberColumn(format="%d", width="small"),
                    "Carbs (g)":   st.column_config.NumberColumn(format="%d", width="small"),
                    "Fat (g)":     st.column_config.NumberColumn(format="%d", width="small"),
                },
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
