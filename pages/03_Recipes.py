# pages/03_Recipes.py
import streamlit as st
from common import ensure_plan_exists
from recipe_db import RECIPE_DB

st.set_page_config(page_title="Recipes — MealPlan Genie", page_icon="📖", layout="wide")
st.title("📖 Recipes")

ensure_plan_exists()
plan = st.session_state.plan

# Only recipes used in the plan
used_names = {r["name"] for meals in plan.values() for r in meals if r}
used = [r for r in RECIPE_DB if r["name"] in used_names]

q = st.text_input("Search recipe name or ingredient", "")
def matches(r):
    if not q: return True
    ql = q.lower()
    if ql in r["name"].lower(): return True
    return any(ql in ing["item"].lower() for ing in r["ingredients"])

filtered = [r for r in used if matches(r)]

for r in filtered:
    with st.expander(r["name"]):
        st.write(f"*Cuisine:* `{r.get('cuisine','')}`  —  *Course:* `{r.get('course','any')}`")
        st.write("**Ingredients:**")
        for ing in r["ingredients"]:
            st.write(f"- {ing.get('qty','')} {ing.get('unit','')} {ing['item']}".strip())
        st.write("**Steps:**")
        for i, step in enumerate(r.get("steps", []), start=1):
            st.write(f"{i}. {step}")
