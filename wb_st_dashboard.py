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
    if is_all_tribes:
        # Sum population for each tribe across selected years
        tribe_ranks = df_filtered.groupby('ST_Name')['Total_P'].sum().nlargest(25).index.tolist()
        df_filtered = df_filtered[df_filtered['ST_Name'].isin(tribe_ranks)]

    # --- Aggregation ---
    agg_cols = ['Year', 'ST_Name', 'Religion_Name']
    if data_level == "District Level" and not is_all_dist: agg_cols.append('District')
    df_agg = df_filtered.groupby(agg_cols)[POPULATION_COLS].sum().reset_index()
    
    # Filter out "All" categories for graphs - keep only individual tribes and religions
    df_graph = df_agg[
        ~(df_agg['ST_Name'].str.contains('All Schedule|All Tribe', case=False, na=False)) &
        ~(df_agg['Religion_Name'].str.contains('All religion', case=False, na=False))
    ].copy()
    
    # For overall stats, get rows where both ST_Name AND Religion_Name are "All"
    df_overall = df_agg[
        (df_agg['ST_Name'].str.contains('All Schedule|All Tribe', case=False, na=False)) &
        (df_agg['Religion_Name'].str.contains('All religion', case=False, na=False))
    ].copy()
    
    # If no "All" rows found (because they were filtered out), compute overall stats from graph data
    if df_overall.empty and not df_graph.empty:
        df_overall = df_graph.groupby('Year')[POPULATION_COLS].sum().reset_index()
        df_overall['ST_Name'] = 'All Tribes'
        df_overall['Religion_Name'] = 'All Religions'

    # ==========================================
    # MAIN PAGE LAYOUT
    # ==========================================
    st.title(f"West Bengal ST Census Trends: 2001 vs 2011 ")
    st.markdown("Use the filters on the left to drill down. Scroll down for detailed analysis.")

    # ------------------------------------------
    # SECTION 1: POPULATION GROWTH
    # ------------------------------------------
    st.markdown("### 1. Population Growth")
    st.caption("Comparison of Total Population counts between 2001 (Light Green) and 2011 (Dark Green) for individual tribes.")
    
    # Metrics from overall stats
    total_2001 = df_overall[df_overall['Year'] == '2001']['Total_P'].sum()
    total_2011 = df_overall[df_overall['Year'] == '2011']['Total_P'].sum()
    growth = ((total_2011 - total_2001) / total_2001 * 100) if total_2001 > 0 else 0
    population_increase = total_2011 - total_2001
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Population 2001", f"{total_2001:,.0f}")
    m2.metric("Population 2011", f"{total_2011:,.0f}")
    m3.metric("Growth Rate (01-11)", f"{growth:+.1f}%")

    # Chart: Side-by-Side Bar Chart (Sorted by 2011 Pop Size) - Individual tribes only
    if not df_graph.empty:
        # Create interval selection for zooming and brushing
        selection = alt.selection_interval(encodings=['x'])
        
        chart_pop = alt.Chart(df_graph).mark_bar().encode(
            x=alt.X('ST_Name', title=None, sort='-y'),
            y=alt.Y('Total_P', title="Population", axis=alt.Axis(format='~s')),
            color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE, legend=alt.Legend(title="Census Year")),
            xOffset=alt.XOffset('Year:N'), 
            tooltip=['ST_Name', 'Year', alt.Tooltip('Total_P', format=',.0f')]
        ).add_selection(
            selection
        ).properties(height=400, title="Population Comparison (Individual Tribes) - Double-click to reset zoom").configure_axis(labelFontSize=11, titleFontSize=13)
        
        # Create zoomed chart
        chart_pop_zoomed = chart_pop.transform_filter(selection)
        
        st.altair_chart(chart_pop_zoomed, use_container_width=True)
    else:
        st.info("No individual tribe data available for this selection.")

    st.markdown("---")

    # ------------------------------------------
    # SECTION 2: GENDER & SEX RATIO
    # ------------------------------------------
    st.markdown("### 2. Gender Balance (Sex Ratio)")
    st.caption("Sex Ratio: Females per 1000 Males for individual tribes. Sorted from **highest to lowest ratio** based on 2011 data.")

    # Prepare Sex Ratio Data from individual tribes only
    df_sex = df_graph.groupby(['Year', 'ST_Name'])[['Total_M', 'Total_F']].sum().reset_index()
    df_sex['Sex_Ratio'] = np.where(df_sex['Total_M'] > 0, (df_sex['Total_F'] / df_sex['Total_M']) * 1000, 0)
    
    # Calculate overall sex ratio for metrics
    overall_sex_2001 = df_overall[df_overall['Year'] == '2001']['Total_F'].sum() / df_overall[df_overall['Year'] == '2001']['Total_M'].sum() * 1000 if df_overall[df_overall['Year'] == '2001']['Total_M'].sum() > 0 else 0
    overall_sex_2011 = df_overall[df_overall['Year'] == '2011']['Total_F'].sum() / df_overall[df_overall['Year'] == '2011']['Total_M'].sum() * 1000 if df_overall[df_overall['Year'] == '2011']['Total_M'].sum() > 0 else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Sex Ratio 2001", f"{overall_sex_2001:.0f}")
    m2.metric("Sex Ratio 2011", f"{overall_sex_2011:.0f}")
    
    # Sort order based on 2011 Sex Ratio (already descending)
    sort_order_sex = df_sex[df_sex['Year'] == '2011'].sort_values('Sex_Ratio', ascending=False)['ST_Name'].tolist()

    # Visualization: Grouped Bar Chart
    if not df_sex.empty:
        # Create interval selection for zooming
        selection_sex = alt.selection_interval(encodings=['x'])
        
        base_sex = alt.Chart(df_sex).encode(
            x=alt.X('ST_Name', title=None, sort=sort_order_sex),
            y=alt.Y('Sex_Ratio', title="Females per 1000 Males", scale=alt.Scale(domain=[0, 2000])),
            tooltip=['ST_Name', 'Year', alt.Tooltip('Sex_Ratio', format='.0f')]
        )
        
        bars_sex = base_sex.mark_bar().encode(
            color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE),
            xOffset=alt.XOffset('Year:N')
        ).add_selection(
            selection_sex
        )

        # Reference line at 950
        ref_line = alt.Chart(pd.DataFrame({'y': [950]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
        
        # Apply selection filter to bars and reference line
        bars_sex_zoomed = bars_sex.transform_filter(selection_sex)
        ref_line_zoomed = ref_line.transform_filter(selection_sex)
        
        st.altair_chart((bars_sex_zoomed + ref_line_zoomed).properties(height=400, title="Sex Ratio Comparison (Individual Tribes) - Double-click to reset zoom"), use_container_width=True)
        st.caption("🔴 Red dotted line represents a reference ratio of 950.")
    else:
        st.info("No individual tribe data available for this selection.")

    st.markdown("---")

    # ------------------------------------------
    # SECTION 3: RELIGION & URBANIZATION
    # ------------------------------------------

    st.markdown("### 3. Religious Composition")
    st.caption("Distribution of religions within individual tribes. Sorted by **Total Population (Descending)**.")
    
    # Calculate sort order based on total population (sum of Total_P) across all years/religions - individual tribes only
    df_total_pop = df_graph.groupby('ST_Name')['Total_P'].sum().reset_index(name='Total_Pop')
    sort_order_rel = df_total_pop.sort_values('Total_Pop', ascending=False)['ST_Name'].tolist()

    # Implementation of Separate Normalized Stacked Bar Charts for 2001 and 2011
    if not df_graph.empty:
        # Create color scheme for religions
        religions = sorted(df_graph['Religion_Name'].unique())
        religion_colors = {}
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, religion in enumerate(religions):
            religion_colors[religion] = base_colors[i % len(base_colors)]
        
        # Chart for 2001
        df_2001 = df_graph[df_graph['Year'] == '2001']
        if not df_2001.empty:
            selection_rel_2001 = alt.selection_interval(encodings=['y'])
            chart_2001 = alt.Chart(df_2001).mark_bar().encode(
                x=alt.X('sum(Total_P)', stack='normalize', axis=alt.Axis(format='%'), title="Share"),
                y=alt.Y('ST_Name', title=None, sort=sort_order_rel),
                color=alt.Color('Religion_Name', title="Religion", scale=alt.Scale(domain=religions, range=[religion_colors[r] for r in religions])),
                tooltip=['ST_Name', 'Religion_Name', alt.Tooltip('sum(Total_P)', format=',.0f')]
            ).add_selection(
                selection_rel_2001
            ).transform_filter(
                selection_rel_2001
            ).properties(height=350, title="Religious Composition 2001 (Individual Tribes) - Double-click to reset zoom")
            
            st.altair_chart(chart_2001, use_container_width=True)
        
        # Chart for 2011
        df_2011 = df_graph[df_graph['Year'] == '2011']
        if not df_2011.empty:
            selection_rel_2011 = alt.selection_interval(encodings=['y'])
            chart_2011 = alt.Chart(df_2011).mark_bar().encode(
                x=alt.X('sum(Total_P)', stack='normalize', axis=alt.Axis(format='%'), title="Share"),
                y=alt.Y('ST_Name', title=None, sort=sort_order_rel),
                color=alt.Color('Religion_Name', title="Religion", scale=alt.Scale(domain=religions, range=[religion_colors[r] for r in religions])),
                tooltip=['ST_Name', 'Religion_Name', alt.Tooltip('sum(Total_P)', format=',.0f')]
            ).add_selection(
                selection_rel_2011
            ).transform_filter(
                selection_rel_2011
            ).properties(height=350, title="Religious Composition 2011 (Individual Tribes) - Double-click to reset zoom")
            
            st.altair_chart(chart_2011, use_container_width=True)
    else:
        st.info("No individual tribe data available for this selection.")

    # ------------------------------------------
    # SECTION 4: URBAN POPULATION TREND
    # ------------------------------------------
    st.markdown("### 4. Urban Population Trend")
    st.caption("Comparison of Urban Population counts between 2001 (Light Green) and 2011 (Dark Green) for individual tribes. Sorted by **2011 Urban Population (Descending)**.")
    
    # Overall urban stats
    urban_2001 = df_overall[df_overall['Year'] == '2001']['Urban_P'].sum()
    urban_2011 = df_overall[df_overall['Year'] == '2011']['Urban_P'].sum()
    urban_growth = ((urban_2011 - urban_2001) / urban_2001 * 100) if urban_2001 > 0 else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Total Urban Population 2001", f"{urban_2001:,.0f}")
    m2.metric("Total Urban Population 2011", f"{urban_2011:,.0f}")
    
    df_urb = df_graph.groupby(['ST_Name', 'Year'])[['Urban_P']].sum().reset_index()
    
    # Sort order based on 2011 Urban Population (descending)
    sort_order_urb = df_urb[df_urb['Year'] == '2011'].sort_values('Urban_P', ascending=False)['ST_Name'].tolist()
    
    # Visualization: Grouped Bar Chart for Urban Population
    if not df_urb.empty:
        # Create interval selection for zooming
        selection_urb = alt.selection_interval(encodings=['x'])
        
        chart_urb = alt.Chart(df_urb).mark_bar().encode(
            x=alt.X('ST_Name', title=None, sort=sort_order_urb),
            y=alt.Y('Urban_P', title="Urban Population", axis=alt.Axis(format='~s')),
            color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE),
            xOffset=alt.XOffset('Year:N'),
            tooltip=['ST_Name', 'Year', alt.Tooltip('Urban_P', format=',.0f')]
        ).add_selection(
            selection_urb
        ).transform_filter(
            selection_urb
        ).properties(height=350, title="Urban Population Trend (Individual Tribes) - Double-click to reset zoom")
        
        st.altair_chart(chart_urb, use_container_width=True)
    else:
        st.info("No individual tribe data available for this selection.")

    st.markdown("---")

    # ------------------------------------------
    # SECTION 5: RURAL POPULATION TREND
    # ------------------------------------------
    st.markdown("### 5. Rural Population Trend")
    st.caption("Comparison of Rural Population counts between 2001 (Light Green) and 2011 (Dark Green) for individual tribes. Sorted by **2011 Rural Population (Descending)**.")
    
    # Overall rural stats
    rural_2001 = df_overall[df_overall['Year'] == '2001']['Rural_P'].sum()
    rural_2011 = df_overall[df_overall['Year'] == '2011']['Rural_P'].sum()
    rural_growth = ((rural_2011 - rural_2001) / rural_2001 * 100) if rural_2001 > 0 else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Total Rural Population 2001", f"{rural_2001:,.0f}")
    m2.metric("Total Rural Population 2011", f"{rural_2011:,.0f}")
    
    df_rural = df_graph.groupby(['ST_Name', 'Year'])[['Rural_P']].sum().reset_index()
    
    # Sort order based on 2011 Rural Population (descending)
    sort_order_rural = df_rural[df_rural['Year'] == '2011'].sort_values('Rural_P', ascending=False)['ST_Name'].tolist()
    
    # Visualization: Grouped Bar Chart for Rural Population
    if not df_rural.empty:
        # Create interval selection for zooming
        selection_rural = alt.selection_interval(encodings=['x'])
        
        chart_rural = alt.Chart(df_rural).mark_bar().encode(
            x=alt.X('ST_Name', title=None, sort=sort_order_rural),
            y=alt.Y('Rural_P', title="Rural Population", axis=alt.Axis(format='~s')),
            color=alt.Color('Year:O', scale=YEAR_COLOR_SCALE),
            xOffset=alt.XOffset('Year:N'),
            tooltip=['ST_Name', 'Year', alt.Tooltip('Rural_P', format=',.0f')]
        ).add_selection(
            selection_rural
        ).transform_filter(
            selection_rural
        ).properties(height=350, title="Rural Population Trend (Individual Tribes) - Double-click to reset zoom")
        
        st.altair_chart(chart_rural, use_container_width=True)
    else:
        st.info("No individual tribe data available for this selection.")

    # --- Data Table ---
    with st.expander("View Underlying Data"):
        st.dataframe(df_filtered)

if __name__ == "__main__":
    main()
