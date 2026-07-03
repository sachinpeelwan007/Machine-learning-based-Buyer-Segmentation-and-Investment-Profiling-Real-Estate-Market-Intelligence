import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Page configuration for a professional wide layout
st.set_page_config(
    page_title="Parcl - Enterprise AI Real Estate Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NATIVE THEME INTEGRATION & MODERN CARD CONTAINER STYLING

st.markdown("""
    <style>
    /* Main Header entry alignment */
    .main-title { 
        font-size: 40px; 
        font-weight: 800; 
        margin-bottom: 2px;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    .subtitle { 
        font-size: 16px; 
        margin-bottom: 30px;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Dynamic Section Container that adapts to Light/Dark themes automatically */
    .section-container {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.05); /* Dynamic transparent tint */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .section-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(128, 128, 128, 0.15);
    }
    
    /* Interactive KPI Cards with subtle background borders */
    .metric-card { 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.3);
        background-color: rgba(128, 128, 128, 0.08);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metric-card h5 {
        margin-bottom: 5px;
        font-weight: 500;
        opacity: 0.8;
    }
    .metric-card h2 {
        margin: 0;
        font-weight: 800;
        font-size: 28px;
    }
    
    .section-header {
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 8px;
        margin-bottom: 20px;
        font-weight: 700;
        font-size: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# DATA LOADING & PROCESSING PIPELINE

@st.cache_data
def load_and_process_data():
    try:
        clients_df = pd.read_csv("clients.csv")
        properties_df = pd.read_csv("properties.csv")
    except FileNotFoundError:
        st.error("Error: 'clients.csv' or 'properties.csv' missing inside project path.")
        st.stop()

    clients_df.drop_duplicates(subset=['client_id'], inplace=True)
    clients_df['satisfaction_score'] = clients_df['satisfaction_score'].fillna(clients_df['satisfaction_score'].median())
    
    clients_df['date_of_birth'] = pd.to_datetime(clients_df['date_of_birth'], format='mixed', errors='coerce')
    clients_df['date_of_birth'] = clients_df['date_of_birth'].fillna(pd.to_datetime('1975-01-01'))
    
    current_year = 2026
    clients_df['age'] = current_year - clients_df['date_of_birth'].dt.year
    
    if 'sale_price' in properties_df.columns:
        properties_df['sale_price_cleaned'] = properties_df['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        client_investment = properties_df.groupby('client_ref').agg(
            total_investment=('sale_price_cleaned', 'sum'),
            properties_bought=('listing_id', 'count')
        ).reset_index()
        clients_df = clients_df.merge(client_investment, left_on='client_id', right_on='client_ref', how='left')
    
    clients_df['total_investment'] = clients_df['total_investment'].fillna(0)
    clients_df['properties_bought'] = clients_df['properties_bought'].fillna(0)

    le = LabelEncoder()
    encoded_df = clients_df.copy()
    categorical_cols = ['client_type', 'region', 'acquisition_purpose', 'referral_channel', 'country', 'gender', 'loan_applied']
    
    for col in categorical_cols:
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        
    features_for_clustering = encoded_df[['age', 'satisfaction_score', 'client_type', 'acquisition_purpose', 'loan_applied']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clients_df['cluster_id'] = kmeans.fit_predict(scaled_features)
    
    cluster_mapping = {
        0: "C1: Global Investors",
        1: "C2: First-Time Buyers",
        2: "C3: Corporate Buyers",
        3: "C4: Luxury Investors"
    }
    clients_df['buyer_segment'] = clients_df['cluster_id'].map(cluster_mapping)
    return clients_df

df = load_and_process_data()

 
# SIDEBAR CONTROLS & SIDE-PANEL FILTERS

st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=75)
st.sidebar.title("Analytics Control Panel")
st.sidebar.write("---")

selected_country = st.sidebar.multiselect("Select Country Target", options=df['country'].unique(), default=df['country'].unique())
selected_region = st.sidebar.multiselect("Select Region Location", options=df['region'].unique(), default=df['region'].unique())
selected_purpose = st.sidebar.multiselect("Acquisition Purpose", options=df['acquisition_purpose'].unique(), default=df['acquisition_purpose'].unique())
selected_client_type = st.sidebar.multiselect("Client Identity Type", options=df['client_type'].unique(), default=df['client_type'].unique())

filtered_df = df[
    (df['country'].isin(selected_country)) &
    (df['region'].isin(selected_region)) &
    (df['acquisition_purpose'].isin(selected_purpose)) &
    (df['client_type'].isin(selected_client_type))
]


# MAIN HUB HEADER DISPLAY

st.markdown('<div class="main-title">Real Estate Market Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Driven Single-Page Executive Dashboard Hub | Developed for Parcl Co. Limited</div>', unsafe_allow_html=True)

# Executive Analytics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><h5>Total Volume Buyers</h5><h2>{len(filtered_df)}</h2></div>', unsafe_allow_html=True)
with m2:
    corp_count = len(filtered_df[filtered_df['client_type'].str.lower().str.contains('company|corporate', na=False)])
    st.markdown(f'<div class="metric-card"><h5>Corporate Entities</h5><h2>{corp_count}</h2></div>', unsafe_allow_html=True)
with m3:
    avg_sat = filtered_df['satisfaction_score'].mean() if len(filtered_df) > 0 else 0
    st.markdown(f'<div class="metric-card"><h5>Avg Satisfaction Index</h5><h2>{avg_sat:.2f} / 5.0</h2></div>', unsafe_allow_html=True)
with m4:
    total_port_val = filtered_df['total_investment'].sum()
    st.markdown(f'<div class="metric-card"><h5>Total Matched Value</h5><h2>${total_port_val:,.2f}</h2></div>', unsafe_allow_html=True)

st.write("##")

# COMPREHENSIVE SINGLE LONG PAGE MODULES

# SECTION 1: BUYER SEGMENTATION OVERVIEW
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">📊 1. Buyer Segmentation Overview</h3>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    segment_counts = filtered_df['buyer_segment'].value_counts()
    if not segment_counts.empty:
        # Dynamic Matplotlib theme adjustment helper
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_alpha(0.0) # Transparent chart background to inherit system theme
        ax.set_facecolor('none')
        colors = ['#1E3A8A', '#10B981', '#F59E0B', '#EF4444']
        
        # Checking if dark theme or light theme text visibility is needed
        text_color = 'grey'
        wedges, texts, autotexts = ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors[:len(segment_counts)])
        for text in texts:
            text.set_color(text_color)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("Filters match no current buyer distribution.")
with col2:
    st.write("##### Recommended Target Framework Matrix")
    st.dataframe(pd.DataFrame({
        "Cluster": ["C1", "C2", "C3", "C4"],
        "Mapped Segment": ["Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"],
        "Strategic Behavioral Trait": ["High asset allocation, pure investment", "Young cohort, intense loan deployment", "Institutional bulk purchase behavior", "Elite satisfaction scores, high ticket properties"]
    }), hide_index=True, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 2: INVESTOR BEHAVIOR DASHBOARD

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">📈 2. Investor Behavior Dashboard</h3>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    st.write("##### Age Metric Span Across Clusters")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        sns.boxplot(data=filtered_df, x='buyer_segment', y='age', palette='Blues', ax=ax)
        plt.xticks(rotation=10)
        ax.tick_params(colors='grey')
        ax.xaxis.label.set_color('grey')
        ax.yaxis.label.set_color('grey')
        st.pyplot(fig)
with b2:
    st.write("##### Strategic Intent (Acquisition Purpose) Breakdown")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        cross_tab = pd.crosstab(filtered_df['buyer_segment'], filtered_df['acquisition_purpose'])
        cross_tab.plot(kind='bar', stacked=True, color=['#1E3A8A', '#60A5FA'], ax=ax)
        plt.xticks(rotation=10)
        ax.tick_params(colors='grey')
        st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 3: GEOGRAPHIC BUYER ANALYSIS

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">🌍 3. Geographic Buyer Analysis</h3>', unsafe_allow_html=True)
geo_col1, geo_col2 = st.columns(2)
with geo_col1:
    st.write("##### Buyer Volume Map by Regional Jurisdictions")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        sns.countplot(data=filtered_df, x='region', hue='buyer_segment', palette='magma', ax=ax)
        ax.tick_params(colors='grey')
        st.pyplot(fig)
with geo_col2:
    st.write("##### Transnational Inbound Matrix Flow")
    if not filtered_df.empty:
        country_matrix = filtered_df.groupby(['country', 'buyer_segment']).size().unstack(fill_value=0)
        st.dataframe(country_matrix, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 4: SEGMENT INSIGHTS PANEL

st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">🔬 4. Segment Insights & ML Statistics Panel</h3>', unsafe_allow_html=True)
if not filtered_df.empty:
    metric_selection = st.selectbox("Select Target Analytics Profile Variable:", ['age', 'satisfaction_score', 'total_investment'])
    summary_table = filtered_df.groupby('buyer_segment')[metric_selection].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
    st.dataframe(summary_table, use_container_width=True)
    
    st.markdown("""
    **💡 Strategic Business Intelligence Takeaways:**
    * **C1/C4 Segments (High-End & Luxury):** Inhe high-value properties pitch karein kyunki inka ticket size bada hai.
    * **C2 Segment (First-Time Buyers):** Inhe targeted financing plans aur easy loan options display karein.
    """)
else:
    st.error("Data parameters are blank for the given values.")
st.markdown('</div>', unsafe_allow_html=True)
