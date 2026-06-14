import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seiteneinstellungen
st.set_page_config(page_title="Ames Housing Dashboard", layout="wide")

# Stil
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Titel
st.title("Ames Housing Dashboard 🏠")
st.write("Analyse der wichtigsten Einflussfaktoren auf den Hauspreis (`SalePrice`).")

# Daten laden
df = pd.read_csv("AmesHousing_cleaned.csv")

# Grundinfos
st.subheader("Datensatzübersicht")
st.write(f"Der Datensatz enthält **{len(df)} Zeilen** und **{len(df.columns)} Spalten**.")
st.dataframe(df.head(10))

# Statistische Übersicht
st.subheader("Statistische Übersicht")
st.write(df.describe())

# -------------------------------
# Visualisierungen
# -------------------------------
st.header("Visualisierungen")

# 1. Histogramm von SalePrice
st.subheader("1. Verteilung von SalePrice")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(df["SalePrice"], bins=30, edgecolor="black")
ax1.set_title("Verteilung von SalePrice")
ax1.set_xlabel("SalePrice")
ax1.set_ylabel("Häufigkeit")
st.pyplot(fig1)

# 2. Boxplot von SalePrice
st.subheader("2. Boxplot von SalePrice")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.boxplot(y=df["SalePrice"], ax=ax2)
ax2.set_title("Boxplot von SalePrice")
st.pyplot(fig2)

# 3. Durchschnittlicher SalePrice nach Overall Qual
st.subheader("3. Durchschnittlicher SalePrice nach Overall Qual")
avg_price_qual = df.groupby("Overall Qual")["SalePrice"].mean()
fig3, ax3 = plt.subplots(figsize=(10, 6))
avg_price_qual.plot(kind="bar", ax=ax3)
ax3.set_title("Durchschnittlicher SalePrice nach Overall Qual")
ax3.set_xlabel("Overall Qual")
ax3.set_ylabel("Durchschnittlicher SalePrice")
st.pyplot(fig3)

# 4. SalePrice nach Wohnflächen-Gruppen
st.subheader("4. Durchschnittlicher SalePrice nach Wohnflächen-Gruppen")
df["Area_Group"] = pd.qcut(df["Gr Liv Area"], 5)
avg_price_area = df.groupby("Area_Group", observed=False)["SalePrice"].mean()
fig4, ax4 = plt.subplots(figsize=(10, 6))
avg_price_area.plot(kind="bar", ax=ax4)
ax4.set_title("Durchschnittlicher SalePrice nach Wohnflächen-Gruppen")
ax4.set_xlabel("Gr Liv Area Gruppen")
ax4.set_ylabel("Durchschnittlicher SalePrice")
plt.xticks(rotation=45)
st.pyplot(fig4)

# 5. Scatter Plot: Gr Liv Area vs SalePrice
st.subheader("5. Gr Liv Area vs. SalePrice")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="Gr Liv Area", y="SalePrice", ax=ax5)
ax5.set_title("Gr Liv Area und SalePrice")
st.pyplot(fig5)

# 6. Top 10 Neighborhoods nach durchschnittlichem SalePrice
st.subheader("6. Top 10 Neighborhoods nach durchschnittlichem SalePrice")
avg_price_neighborhood = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
fig6, ax6 = plt.subplots(figsize=(10, 6))
avg_price_neighborhood.head(10).plot(kind="bar", ax=ax6)
ax6.set_title("Top 10 Neighborhoods nach durchschnittlichem SalePrice")
ax6.set_xlabel("Neighborhood")
ax6.set_ylabel("Durchschnittlicher SalePrice")
plt.xticks(rotation=45)
st.pyplot(fig6)

# 7. Boxplot: SalePrice nach Central Air
st.subheader("7. SalePrice nach Central Air")
fig7, ax7 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Central Air", y="SalePrice", ax=ax7)
ax7.set_title("SalePrice nach Central Air")
st.pyplot(fig7)

# 8. Korrelationsmatrix
st.subheader("8. Korrelationsmatrix wichtiger numerischer Variablen")
top_num_cols = [
    "SalePrice",
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "Year Built",
    "Full Bath",
    "Fireplaces"
]

corr_matrix = df[top_num_cols].corr()

fig8, ax8 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax8)
ax8.set_title("Korrelationsmatrix")
st.pyplot(fig8)