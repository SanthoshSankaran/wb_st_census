import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Census Analysis 2001-2011")

# --- Constants & Mappings ---
COLUMN_MAP = {
    'Year': 'Year',
    'State': 'State',
    'District': 'District',
    'ST Name': 'ST_Name', 
    'Religion': 'Religion_Name', 
    'Persons': 'Total_P', 
    'Male': 'Total_M', 
    'Female': 'Total_F', 
    'Persons (Rural)': 'Rural_P', 
    'Male (Rural)': 'Rural_M', 
    'Female (Rural)': 'Rural_F',
    'Persons (Urban)': 'Urban_P', 
    'Male (Urban)': 'Urban_M', 
    'Female (Urban)': 'Urban_F'
}

POPULATION_COLS = ['Total_P', 'Total_M', 'Total_F', 'Rural_P', 'Rural_M', 'Rural_F', 'Urban_P', 'Urban_M', 'Urban_F']

# Light Green (2001) -> Dark Green (2011)
YEAR_COLOR_SCALE = alt.Scale(domain=['2001', '2011'], range=['#81C784', '#1B5E20'])

# --- Data Loading ---
@st.cache_data
def load_single_file(filepath, level_tag):
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
        if 'District' not in df.columns: df['District'] = 'All Districts'
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.replace(r'[",\s]', '', regex=True)
        else:
            df['Year'] = 'Unknown'
        df['Level'] = level_tag
        
        for col in [c for c in POPULATION_COLS if c in df.columns]:
            df[col] = df[col].astype(str).str.replace(r'[",\s]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        if 'ST_Name' in df.columns:
            df = df[df['ST_Name'].notna()]
        if 'Religion_Name' in df.columns:
            df['Religion_Name'] = df['Religion_Name'].astype(str).str.replace('Budhists', 'Buddhists', case=False)
            
        return df.drop(columns=['ST Code', 'ST_Code'], errors='ignore')
    except: return pd.DataFrame()

@st.cache_data
def load_all_datasets():
    base_path = "./"
    files = [
        (base_path + "2011 WB State ST Data.csv", "State"),
        (base_path + "2001 WB State ST Data.csv", "State"),
        (base_path + "2011 WB District ST Data.csv", "District"),
        (base_path + "2001 WB District ST Data.csv", "District")
    ]
    dfs = [load_single_file(f, l) for f, l in files]
    
    # Filter out empty dataframes and check if the resulting list is empty before concatenation
    non_empty_dfs = [d for d in dfs if not d.empty]
    
    if not non_empty_dfs:
        # If the list is empty (no files loaded successfully), return an empty DataFrame
        return pd.DataFrame()
    
    return pd.concat(non_empty_dfs, ignore_index=True)

# --- Main Dashboard ---
def main():
    # Load Data
    df = load_all_datasets()
    if df.empty:
        st.error("Data not found.")
        return

    # --- Sidebar ---
    st.sidebar.title("Filters")
    data_level = st.sidebar.radio("Level", ["State Level", "District Level"])
    df_lvl = df[df['Level'] == 'State'] if data_level == "State Level" else df[df['Level'] == 'District']
    
    def filter_box(label, col):
        opts = sorted(df_lvl[col].astype(str).unique())
        sel = st.sidebar.selectbox(label, ["All"] + opts)
        return opts if sel == "All" else [sel], sel == "All"

    if data_level == "District Level":
        sel_dist, is_all_dist = filter_box("District", "District")
        df_lvl = df_lvl[df_lvl['District'].isin(sel_dist)]
    
    sel_years, is_all_years = filter_box("Year", "Year")
    sel_tribes, is_all_tribes = filter_box("Tribe", "ST_Name")
    sel_rel, is_all_rel = filter_box("Religion", "Religion_Name")

    df_filtered = df_lvl[
        df_lvl['Year'].isin(sel_years) & 
        df_lvl['ST_Name'].isin(sel_tribes) & 
        df_lvl['Religion_Name'].isin(sel_rel)
    ]

    if df_filtered.empty:
        st.warning("No data matches filters.")
        return

    # --- Logic for "Top N" Display ---
    # If viewing "All" tribes, only show Top 15 by population to keep charts readable
    if is_all_tribes:
        # Sum population for each tribe across selected years
        tribe_ranks = df_filtered.groupby('ST_Name')['Total_P'].sum().nlargest(15).index.tolist()
        df_filtered = df_filtered[df_filtered['ST_Name'].isin(tribe_ranks)]

    # --- Aggregation ---
    agg_cols = ['Year', 'ST_Name', 'Religion_Name']
    if data_level == "District Level" and not is_all_dist: agg_cols.append('District')
    df_agg = df_filtered.groupby(agg_cols)[POPULATION_COLS].sum().reset_index()

    # ==========================================
    # MAIN PAGE LAYOUT
    # ==========================================
    st.title(f"West Bengal ST Census Trends: 2001 vs 2011 ")
    st.markdown("Use the filters on the left to drill down. Scroll down for detailed analysis.")

    # ------------------------------------------
    # SECTION 1: POPULATION GROWTH
    # ------------------------------------------
    st.markdown("### 1. Population Growth")
    st.caption("Comparison of Total Population counts between 2001 (Light Green) and 2011 (Dark Green).")
    
    # Metrics
    total_2001 = df_agg[df_agg['Year'] == '2001']['Total_P'].sum()
    total_2011 = df_agg[df_agg['Year'] == '2011']['Total_P'].sum()
    growth = ((total_2011 - total_2001) / total_2001 * 100) if total_2001 > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Population (Selected)", f"{df_agg['Total_P'].sum():,.0f}")
    m2.metric("Growth Rate (01-11)", f"{growth:+.1f}%")
    m3.metric("Dataset coverage", "Top 15 Tribes" if is_all_tribes else "Selected Tribes")

    # Chart: Side-by-Side Bar Chart (Sorted by 2011 Pop Size)
    chart_pop = alt.Chart(df_agg).mark_bar().encode(
        x=alt.X('ST_Name', title=None, sort='-y'),
        y=alt.Y('Total_P', title="Population", axis=alt.Axis(format='~s')),
        color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE, legend=alt.Legend(title="Census Year")),
        xOffset=alt.XOffset('Year:N'), 
        tooltip=['ST_Name', 'Year', alt.Tooltip('Total_P', format=',.0f')]
    ).properties(height=400, title="Population Comparison (Sorted by Pop Size)").configure_axis(labelFontSize=11, titleFontSize=13)
    
    st.altair_chart(chart_pop, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------
    # SECTION 2: GENDER & SEX RATIO
    # ------------------------------------------
    st.markdown("### 2. Gender Balance (Sex Ratio)")
    st.caption("Sex Ratio: Females per 1000 Males. Sorted from **highest to lowest ratio** based on 2011 data.")

    # Prepare Sex Ratio Data
    df_sex = df_agg.groupby(['Year', 'ST_Name'])[['Total_M', 'Total_F']].sum().reset_index()
    df_sex['Sex_Ratio'] = np.where(df_sex['Total_M'] > 0, (df_sex['Total_F'] / df_sex['Total_M']) * 1000, 0)
    
    # Sort order based on 2011 Sex Ratio (already descending)
    sort_order_sex = df_sex[df_sex['Year'] == '2011'].sort_values('Sex_Ratio', ascending=False)['ST_Name'].tolist()

    # Visualization: Grouped Bar Chart
    base_sex = alt.Chart(df_sex).encode(
        x=alt.X('ST_Name', title=None, sort=sort_order_sex),
        y=alt.Y('Sex_Ratio', title="Females per 1000 Males", scale=alt.Scale(domain=[0, 2000])),
        tooltip=['ST_Name', 'Year', alt.Tooltip('Sex_Ratio', format='.0f')]
    )
    
    bars_sex = base_sex.mark_bar().encode(
        color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE),
        xOffset=alt.XOffset('Year:N')
    )

    # Reference line at 950
    ref_line = alt.Chart(pd.DataFrame({'y': [950]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
    
    st.altair_chart((bars_sex + ref_line).properties(height=400, title="Sex Ratio Comparison (Sorted by 2011 Ratio)"), use_container_width=True)
    st.caption("🔴 Red dotted line represents a reference ratio of 950.")

    st.markdown("---")

    # ------------------------------------------
    # SECTION 3: RELIGION & URBANIZATION
    # ------------------------------------------

    st.markdown("### 3. Religious Composition")
    st.caption("Distribution of religions within the tribes, separated by year. Sorted by **Total Population (Descending)**.")
    
    # Calculate sort order based on total population (sum of Total_P) across all years/religions
    df_total_pop = df_agg.groupby('ST_Name')['Total_P'].sum().reset_index(name='Total_Pop')
    sort_order_rel = df_total_pop.sort_values('Total_Pop', ascending=False)['ST_Name'].tolist()

    # Implementation of Faceted Normalized Stacked Bar Chart
    chart_rel = alt.Chart(df_agg).mark_bar().encode(
        x=alt.X('sum(Total_P)', stack='normalize', axis=alt.Axis(format='%'), title="Share"),
        # Sort tribes by total population
        y=alt.Y('ST_Name', title=None, sort=sort_order_rel), 
        # Use 'Year' to create side-by-side facets
        column=alt.Column('Year:O', title="Census Year"),
        color=alt.Color('Religion_Name', title="Religion", scale=alt.Scale(scheme='tableau10')),
        tooltip=['ST_Name', 'Year', 'Religion_Name', alt.Tooltip('sum(Total_P)', format=',.0f')]
    ).properties(height=350).configure_facet(
        spacing=15 # Add spacing between the two year columns
    ).configure_header(
        titleFontSize=14,
        labelFontSize=12
    )
    st.altair_chart(chart_rel, use_container_width=True)

    # ------------------------------------------
    # SECTION 4: URBANIZATION
    # ------------------------------------------
    st.markdown("### 4. Urbanization")
    st.caption("Proportion of population living in Urban areas. Sorted by **2011 Urban Share (Descending)**.")
    
    df_urb = df_filtered.groupby(['ST_Name', 'Year'])[['Rural_P', 'Urban_P']].sum().reset_index()
    df_urb['Total'] = df_urb['Rural_P'] + df_urb['Urban_P']
    df_urb['Urban_Share'] = np.where(df_urb['Total'] > 0, (df_urb['Urban_P'] / df_urb['Total']), 0)
    
    # Sort order based on 2011 Urban Share (already descending)
    sort_order_urb = df_urb[df_urb['Year'] == '2011'].sort_values('Urban_Share', ascending=False)['ST_Name'].tolist()
    
    # Visualization: Grouped Bar Chart
    chart_urb = alt.Chart(df_urb).mark_bar().encode(
        x=alt.X('ST_Name', title=None, sort=sort_order_urb),
        y=alt.Y('Urban_Share', axis=alt.Axis(format='%'), title="Urban Percentage"),
        color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE),
        xOffset=alt.XOffset('Year:N'),
        tooltip=['ST_Name', 'Year', alt.Tooltip('Urban_Share', format='.1%')]
    ).properties(height=350)
    st.altair_chart(chart_urb, use_container_width=True)

    # --- Data Table ---
    with st.expander("View Underlying Data"):
        st.dataframe(df_filtered)

if __name__ == "__main__":
    main()
