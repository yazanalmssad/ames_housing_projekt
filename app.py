
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------------
# Seiteneinstellungen
# -------------------------------
st.set_page_config(page_title="Ames Housing Präsentation", layout="wide")

# Stil
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# -------------------------------
# Daten laden
# -------------------------------
raw_df = pd.read_csv("AmesHousing.csv")
df = pd.read_csv("AmesHousing_cleaned.csv")

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

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Bereich auswählen:",
    [
        "Start",
        "Datensatz",
        "Data Cleaning",
        "Wichtige Einflussfaktoren",
        "Visualisierungen",
        "Fazit"
    ]
)

# -------------------------------
# START
# -------------------------------
if section == "Start":
    st.title("Analyse von Einflussfaktoren auf Hauspreise im Ames-Housing-Datensatz 🏠")
    st.markdown("""
    In dieser Anwendung präsentiere ich mein Projekt zur Frage,  
    **welche Merkmale den Hauspreis am stärksten beeinflussen**.

    Die Zielvariable ist **„SalePrice“**.
    """)
    st.markdown("""
    **Inhalt dieser App:**
    - Datensatz
    - Data Cleaning
    - Wichtige Einflussfaktoren
    - Visualisierungen
    - Fazit
    """)

# -------------------------------
# DATENSATZ
# -------------------------------
elif section == "Datensatz":
    st.header("1. Datensatz")

    st.markdown("""
    - Datensatz: **Ames Housing Dataset**
    - Quelle: **Kaggle**
    - Reale Verkaufsdaten von Wohnimmobilien in **Ames, Iowa, USA**
    - Zielvariable: **„SalePrice“**
    """)

    st.subheader("Datensatzübersicht (Rohdatensatz)")
    st.write(f"Der Rohdatensatz enthält **{len(raw_df)} Zeilen** und **{len(raw_df.columns)} Spalten**.")
    st.dataframe(raw_df.head(10))

    st.subheader("Statistische Übersicht (Rohdatensatz)")
    st.write(raw_df.describe())

# -------------------------------
# DATA CLEANING
# -------------------------------
elif section == "Data Cleaning":
    st.header("2. Data Cleaning")

    st.markdown("""
    Meine Data-Cleaning-Pipeline bestand aus folgenden Schritten:

    1. Überblick über den Rohdatensatz  
    2. Numerische Spalten mit **„SalePrice“** vergleichen  
    3. Weniger relevante numerische Spalten entfernen  
    4. Kategoriale Spalten systematisch prüfen  
    5. Boxplots für wichtige kategoriale Variablen  
    6. Schwächere kategoriale Spalten entfernen  
    7. Fehlende Werte befüllen  
    8. Duplikat entfernen  
    9. Datentypen anpassen  
    10. Bereinigten Datensatz speichern  
    """)

    st.subheader("2.1 Überblick über den Rohdatensatz")
    st.markdown(f"""
    - ursprünglicher Datensatz: **{len(raw_df)} Zeilen, {len(raw_df.columns)} Spalten**
    - Probleme:
      - fehlende Werte
      - viele kategoriale Spalten
      - ein Duplikat
    """)

    st.subheader("2.2 Numerische Spalten und Korrelation mit „SalePrice“")
    numeric_df = raw_df.select_dtypes(include=["int64", "float64"])
    corr_with_price = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

    st.markdown("**Alle Korrelationen mit „SalePrice“:**")
    st.dataframe(
        corr_with_price.reset_index().rename(
            columns={"index": "Spalte", "SalePrice": "Korrelation"}
        )
    )

    corr_plot = corr_with_price.drop("SalePrice")
    fig_corr, ax_corr = plt.subplots(figsize=(14, 8))
    corr_plot.plot(kind="bar", ax=ax_corr)
    ax_corr.set_title("Korrelation numerischer Spalten mit SalePrice")
    ax_corr.set_xlabel("Spalten")
    ax_corr.set_ylabel("Korrelation mit SalePrice")
    plt.xticks(rotation=90)
    st.pyplot(fig_corr)

    st.subheader("2.3 Kategoriale Spalten systematisch prüfen")
    categorical_cols = raw_df.select_dtypes(include=["object", "category"]).columns

    cat_results = []
    for col in categorical_cols:
        grouped = raw_df.groupby(col)["SalePrice"]
        means = grouped.mean()
        counts = grouped.size()

        cat_results.append({
            "Spalte": col,
            "Anzahl_Kategorien": raw_df[col].nunique(),
            "Kleinste_Gruppe": counts.min(),
            "Groesste_Gruppe": counts.max(),
            "Min_Durchschnittspreis": round(means.min(), 0),
            "Max_Durchschnittspreis": round(means.max(), 0),
            "Differenz": round(means.max() - means.min(), 0),
            "Std_der_Gruppenmittel": round(means.std(), 0)
        })

    cat_results_df = pd.DataFrame(cat_results).sort_values(
        by=["Std_der_Gruppenmittel", "Differenz"],
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(cat_results_df)

    st.subheader("2.4 Boxplots wichtiger kategorialer Variablen")

    fig_box1, ax_box1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=raw_df, x="Central Air", y="SalePrice", ax=ax_box1)
    ax_box1.set_title("SalePrice nach Central Air")
    st.pyplot(fig_box1)

    fig_box2, ax_box2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=raw_df, x="Kitchen Qual", y="SalePrice", ax=ax_box2)
    ax_box2.set_title("SalePrice nach Kitchen Qual")
    st.pyplot(fig_box2)

    st.subheader("2.5 Fehlende Werte im Rohdatensatz")
    missing_values = raw_df.isnull().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    st.dataframe(
        missing_values.reset_index().rename(
            columns={"index": "Spalte", 0: "Fehlende Werte"}
        )
    )

    st.subheader("2.6 Ergebnis des Cleanings")
    st.markdown(f"""
    - finaler Datensatz: **{len(df)} Zeilen, {len(df.columns)} Spalten**
    - keine fehlenden Werte mehr
    - kategoriale Spalten in **„category“** umgewandelt
    - final gespeichert als **„AmesHousing_cleaned.csv“**
    """)

# -------------------------------
# WICHTIGE EINFLUSSFAKTOREN
# -------------------------------
elif section == "Wichtige Einflussfaktoren":
    st.header("3. Wichtige Einflussfaktoren auf den Hauspreis")

    st.subheader("Wichtige numerische Variablen")
    st.markdown("""
    - **„Overall Qual“** = allgemeine Hausqualität  
    - **„Gr Liv Area“** = oberirdische Wohnfläche  
    - **„Garage Cars“** = Anzahl der Stellplätze in der Garage  
    - **„Garage Area“** = Garagenfläche  
    - **„Total Bsmt SF“** = gesamte Kellerfläche  
    - **„1st Flr SF“** = Fläche des Erdgeschosses  
    - **„Year Built“** = Baujahr  
    """)

    st.subheader("Wichtige kategoriale Variablen")
    st.markdown("""
    - **„Neighborhood“** = Stadtteil / Lage  
    - **„Exter Qual“** = Qualität des Außenbereichs / Außenmaterials  
    - **„Kitchen Qual“** = Qualität der Küche  
    - **„Central Air“** = zentrale Klimaanlage ja/nein  
    - **„Bsmt Qual“** = Qualität des Kellers  
    """)

# -------------------------------
# VISUALISIERUNGEN
# -------------------------------
elif section == "Visualisierungen":
    st.header("4. Visualisierungen")

    st.subheader("4.1 Verteilung von „SalePrice“")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(df["SalePrice"], bins=30, edgecolor="black")
    ax1.set_title("Verteilung von SalePrice")
    ax1.set_xlabel("SalePrice")
    ax1.set_ylabel("Häufigkeit")
    st.pyplot(fig1)

    st.markdown("""
    **Interpretation:**  
    Die Verteilung zeigt, wie sich die Verkaufspreise im Datensatz verteilen.
    """)

    st.subheader("4.2 Boxplot von „SalePrice“")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=df["SalePrice"], ax=ax2)
    ax2.set_title("Boxplot von SalePrice")
    st.pyplot(fig2)

    st.subheader("4.3 Durchschnittlicher „SalePrice“ nach „Overall Qual“")
    avg_price_qual = df.groupby("Overall Qual")["SalePrice"].mean()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    avg_price_qual.plot(kind="bar", ax=ax3)
    ax3.set_title("Durchschnittlicher SalePrice nach Overall Qual")
    ax3.set_xlabel("Overall Qual")
    ax3.set_ylabel("Durchschnittlicher SalePrice")
    st.pyplot(fig3)

    st.subheader("4.4 Durchschnittlicher „SalePrice“ nach Wohnflächen-Gruppen")
    plot_df = df.copy()
    plot_df["Area_Group"] = pd.qcut(plot_df["Gr Liv Area"], 5)
    avg_price_area = plot_df.groupby("Area_Group", observed=False)["SalePrice"].mean()

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    avg_price_area.plot(kind="bar", ax=ax4)
    ax4.set_title("Durchschnittlicher SalePrice nach Wohnflächen-Gruppen")
    ax4.set_xlabel("Gr Liv Area Gruppen")
    ax4.set_ylabel("Durchschnittlicher SalePrice")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.subheader("4.5 Zusammenhang zwischen „Gr Liv Area“ und „SalePrice“")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="Gr Liv Area", y="SalePrice", ax=ax5)
    ax5.set_title("Gr Liv Area und SalePrice")
    st.pyplot(fig5)

    st.subheader("4.6 Top-10-„Neighborhoods“ nach durchschnittlichem „SalePrice“")
    avg_price_neighborhood = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    avg_price_neighborhood.head(10).plot(kind="bar", ax=ax6)
    ax6.set_title("Top 10 Neighborhoods nach durchschnittlichem SalePrice")
    ax6.set_xlabel("Neighborhood")
    ax6.set_ylabel("Durchschnittlicher SalePrice")
    plt.xticks(rotation=45)
    st.pyplot(fig6)

    st.subheader("4.7 „SalePrice“ nach „Central Air“")
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Central Air", y="SalePrice", ax=ax7)
    ax7.set_title("SalePrice nach Central Air")
    st.pyplot(fig7)

    st.subheader("4.8 Korrelationsmatrix wichtiger numerischer Variablen")
    corr_matrix = df[top_num_cols].corr()

    fig8, ax8 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax8)
    ax8.set_title("Korrelationsmatrix")
    st.pyplot(fig8)

    st.subheader("4.9 Interaktiver Scatter Plot")
    fig9 = px.scatter(
        df,
        x="Gr Liv Area",
        y="SalePrice",
        color="Overall Qual",
        size="Garage Area",
        hover_data=["Neighborhood", "Year Built", "Garage Cars"],
        title="Interaktiver Scatter Plot: Gr Liv Area vs. SalePrice"
    )
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    Dieser interaktive Scatter Plot zeigt den Zusammenhang zwischen Wohnfläche und Verkaufspreis.  
    Zusätzlich werden Hausqualität, Garagenfläche und weitere Informationen beim Hovern sichtbar.
    """)

    st.subheader("4.10 Interaktiver Box Plot")
    fig10 = px.box(
        df,
        x="Exter Qual",
        y="SalePrice",
        color="Exter Qual",
        title="Interaktiver Box Plot: SalePrice nach Exter Qual"
    )
    st.plotly_chart(fig10, use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    Der interaktive Box Plot zeigt, dass sich der Verkaufspreis auch nach der Außenqualität deutlich unterscheidet.  
    Je besser die Außenqualität, desto höher ist im Durchschnitt häufig auch der Hauspreis.
    """)

# -------------------------------
# FAZIT
# -------------------------------
elif section == "Fazit":
    st.header("5. Fazit")

    st.markdown("""
    Die wichtigsten Einflussfaktoren auf den Hauspreis waren in meinem Projekt vor allem:

    - **Hausqualität**
    - **Wohnfläche**
    - **Garage**
    - **Kellerfläche**
    - **Lage**
    - **Baujahr**
    - **Ausstattungsmerkmale**
        - **„Central Air“** = zentrale Klimaanlage
        - **„Kitchen Qual“** = Qualität der Küche
        - **„Exter Qual“** = Qualität des Außenbereichs / Außenmaterials
        - **„Bsmt Qual“** = Qualität des Kellers
        - **„Garage Finish“** = Ausbauzustand der Garage
        - **„Garage Type“** = Art der Garage
        - **„Fireplaces“** = Anzahl der Kamine
        - **„Mas Vnr Area“** = Fläche der Mauerverblendung         

    """)

