import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# Load Data
df = pd.read_excel("Data_Kinerja_Karyawan.xlsx")
df['Periode'] = pd.to_datetime(df['Periode'], format='%b-%y')

# Feature Engineering
df['Gap'] = df['Actual'] - df['Target Nilai']
df['Achievement (%)'] = (df['Actual'] / df['Target Nilai']) * 100
df['Status'] = np.where(df['Actual'] >= df['Target Nilai'], 'Tercapai', 'Tidak Tercapai')

# Clustering
features = df.groupby('Nama Karyawan').agg({
    'Achievement (%)': 'mean',
    'Gap': 'mean',
    'Status': lambda x: (x == 'Tercapai').sum()
}).rename(columns={'Status': 'Tugas_Tercapai'})
scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
features['Cluster'] = kmeans.labels_
df = df.merge(features['Cluster'], on='Nama Karyawan', how='left')

# Inkonsistensi
df = df.sort_values(['Nama Karyawan', 'Periode'])
df['Rolling_Std'] = df.groupby('Nama Karyawan')['Achievement (%)'].transform(lambda x: x.rolling(window=3).std())
df['Inkonsisten'] = df['Rolling_Std'] > 10

# Streamlit UI
st.title("People Analytics Dashboard")
st.markdown(
    "_Analisis Segmentasi dan Konsistensi Kinerja Karyawan: Penerapan K-Means Clustering dan Rolling Standard Deviation dalam People Analytics Berbasis Streamlit_"
)

# Profil Tiap Cluster
profil_cluster = features.groupby('Cluster').agg({
    'Achievement (%)': 'mean',
    'Gap': 'mean',
    'Tugas_Tercapai': 'mean'
}).round(2)

profil_cluster = profil_cluster.rename(columns={
    'Achievement (%)': 'Rata-rata Achievement (%)',
    'Gap': 'Rata-rata Gap',
    'Tugas_Tercapai': 'Rata-rata Tugas Tercapai'
}).reset_index()

# Filter
cluster_list = ['Semua'] + sorted(df['Cluster'].unique().tolist())
cluster_filter = st.sidebar.selectbox("Filter Cluster", cluster_list)
df_filtered = df if cluster_filter == 'Semua' else df[df['Cluster'] == cluster_filter]

# KPI
st.subheader("📝 KPI Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Achievement", f"{df_filtered['Achievement (%)'].mean():.2f}")
col2.metric("Tercapai", (df_filtered['Status'] == 'Tercapai').sum())
col3.metric("Tidak Tercapai", (df_filtered['Status'] == 'Tidak Tercapai').sum())
col4.metric("Inkonsisten", df_filtered['Inkonsisten'].sum())

# 1. STACKED BAR CHART: Status Tugas Karyawan (Tercapai vs Tidak)
st.subheader("1️⃣ Distribusi Capaian Tugas Karyawan")

pivot_status = pd.crosstab(df['Nama Karyawan'], df['Status'])
pivot_status = pivot_status.sort_values(by='Tercapai', ascending=False)

warna_tercapai = '#1f77b4'   # Biru tua
warna_tidak = '#aec7e8'      # Biru muda

fig1, ax1 = plt.subplots(figsize=(12, 6))
bar1 = ax1.bar(pivot_status.index, pivot_status['Tercapai'], color=warna_tercapai, label='Tercapai')
bar2 = ax1.bar(pivot_status.index, pivot_status['Tidak Tercapai'],
               bottom=pivot_status['Tercapai'], color=warna_tidak, label='Tidak Tercapai')

for i, (t, tt) in enumerate(zip(pivot_status['Tercapai'], pivot_status['Tidak Tercapai'])):
    ax1.text(i, t / 2, str(t), ha='center', va='center', color='white', fontsize=9, weight='bold')
    ax1.text(i, t + tt / 2, str(tt), ha='center', va='center', color='black', fontsize=9)

ax1.set_ylabel("Jumlah Tugas")
plt.xticks()
ax1.legend()
plt.tight_layout()
st.pyplot(fig1)

# 2. HORIZONTAL BAR CHART: Rata-rata Performa per Indikator
st.subheader("2️⃣ Distribusi Capaian Performa Jobdesk")

indikator_perf = df.groupby('Jobdesk')['Achievement (%)'].mean().sort_values(ascending=False)
colors2 = sns.color_palette("Blues", n_colors=len(indikator_perf))[::-1]

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.barh(indikator_perf.index, indikator_perf.values, color=colors2)
ax2.bar_label(bars, labels=[f"{val:.1f}%" for val in indikator_perf.values],
              label_type='edge', fontsize=10, padding=3)

ax2.set_xlabel('Rata-rata Capaian (%)')
plt.gca().invert_yaxis()
plt.tight_layout()
st.pyplot(fig2)

# 3. Heatmapp Capaian Aktual
st.subheader("3️⃣ Heatmap Capaian Aktual Bulanan per Karyawan")

# Data preprocessing
df['Periode'] = pd.to_datetime(df['Periode'])
df['Bulan'] = df['Periode'].dt.strftime('%b-%y')
pivot = df.pivot_table(
    index='Nama Karyawan',
    columns='Bulan',
    values='Achievement (%)',
    aggfunc='mean'
)

# Urutkan kolom bulan berdasarkan waktu
bulan_order = pd.to_datetime(pivot.columns, format='%b-%y').sort_values()
pivot = pivot[bulan_order.strftime('%b-%y')]

# Plot heatmap
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".0f",
    cmap='Blues',
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Achievement (%)'},
    ax=ax3
)

ax3.set_xlabel('Periode Bulan')
ax3.set_ylabel('Nama Karyawan')
plt.xticks()
plt.tight_layout()

# Tampilkan plot di Streamlit
st.pyplot(fig3)

# 4. Profil Cluster
st.subheader("🔖 Profil Tiap Cluster")
st.dataframe(profil_cluster)

# 5. Pie Chart - Cluster
st.subheader("🧮 Distribusi Cluster")
cluster_counts = df_filtered['Cluster'].value_counts()
colors_pie = sns.color_palette("Blues", n_colors=len(cluster_counts))

fig4, ax4 = plt.subplots()
ax4.pie(
    cluster_counts,
    labels=cluster_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_pie
)
ax4.axis('equal')
st.pyplot(fig4)


# 6. Waterfall Chart - Inkonsisten
st.subheader("📈 Karyawan dengan Performa Inkonsisten")
inkon = df_filtered[df_filtered['Inkonsisten']].groupby('Nama Karyawan')['Rolling_Std'].mean().sort_values(ascending=False)

if not inkon.empty:
    fig4 = go.Figure(go.Waterfall(
        name="Rolling Std",
        orientation="v",
        x=inkon.index,
        measure=["relative"] * len(inkon),
        y=inkon.values,
        textposition="outside",
        text=[f"{v:.1f}" for v in inkon.values],
        increasing=dict(marker=dict(color="rgb(54, 100, 139)"))  # biru tua
    ))
    fig4.update_layout(
        title="",
        waterfallgroupgap=0.5
    )
    st.plotly_chart(fig4)
else:
    st.info("Tidak ada karyawan inkonsisten dalam filter ini.")


# Full Table
st.subheader("📚 Data Lengkap")
st.dataframe(df_filtered)