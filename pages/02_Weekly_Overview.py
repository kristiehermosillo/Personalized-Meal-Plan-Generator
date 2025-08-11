# pages/02_Weekly_Overview.py
import streamlit as st
from common import ensure_plan_exists, plan_to_dataframe, consolidate_shopping_list, get_day_slots

st.set_page_config(page_title="Weekly Overview — MealPlan Genie", page_icon="🗓️", layout="wide")
st.title("🗓️ Weekly Overview")

ensure_plan_exists()
plan = st.session_state.plan

# Tables
meals_per_day = len(plan[1]) if plan.get(1) else 3
df_plan = plan_to_dataframe(plan, meals_per_day)
c1, c2 = st.columns([0.6, 0.4])
with c1:
    st.subheader("Your plan")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)
    if st.session_state.is_premium:
        day_summary = df_plan.groupby("day")[["calories","protein_g","carbs_g","fat_g"]].sum().reset_index()
        st.subheader("Daily totals")
        st.dataframe(day_summary, use_container_width=True, hide_index=True)
with c2:
    st.subheader("Shopping list")
    df_shop = consolidate_shopping_list(plan)
    st.dataframe(df_shop, use_container_width=True, hide_index=True)

# Downloads (CSV/PDF from Home if you want; or re-add here)

st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=True):
    st.switch_page("app.py")
