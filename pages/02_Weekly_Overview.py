# pages/02_Weekly_Overview.py
import streamlit as st
import pandas as pd
from common import plan_to_dataframe, consolidate_shopping_list, parse_pantry_text, split_shopping_by_pantry

st.set_page_config(page_title="Weekly Overview", page_icon="🗓️", layout="wide")
st.title("🗓️ Weekly Overview")

plan = st.session_state.get("plan", {})

# Guard: stop politely if no plan
if not plan or not list(plan.keys()):
    st.info("No plan yet. Go back to **Home** and click **Generate / Regenerate plan**.")
    st.stop()

# Build dataframes safely
df_plan = plan_to_dataframe(plan, len(plan.get(1, [])) or 3)
if df_plan is None or df_plan.empty or "day" not in df_plan.columns:
    st.info("Plan exists but has no rows yet. Generate a new plan on **Home**.")
    st.stop()

df_shop = consolidate_shopping_list(plan)

c1, c2 = st.columns([0.6, 0.4])

with c1:
    st.subheader("Your plan")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    if st.session_state.get("is_premium", False):
        needed_cols = {"calories", "protein_g", "carbs_g", "fat_g"}
        if needed_cols.issubset(set(df_plan.columns)):
            day_summary = (
                df_plan.groupby("day")[["calories", "protein_g", "carbs_g", "fat_g"]]
                .sum()
                .reset_index()
            )
            st.subheader("Daily totals")
            st.dataframe(day_summary, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Shopping list")

    pantry_items = parse_pantry_text(st.session_state.get("pantry_text", ""))
    annotate = st.session_state.get("show_pantry_note", False)
    need_df, have_df = split_shopping_by_pantry(df_shop, pantry_items, annotate_at_bottom=annotate)

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
        data=df_plan.to_csv(index=False).encode(),
        file_name="mealplan.csv",
        mime="text/csv",
    )
with dl2:
    st.download_button(
        "Download Shopping List (CSV)",
        data=df_shop.to_csv(index=False).encode(),
        file_name="shopping_list.csv",
        mime="text/csv",
    )
