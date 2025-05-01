import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

# --- Fonctions mathématiques ---
def f(x):
    return np.exp(-x**2)

def f_double_prime(x):
    return (4 * x**2 - 2) * np.exp(-x**2)

def f_fourth_prime(x):
    return (16 * x**4 - 48 * x**2 + 12) * np.exp(-x**2)

def Gauss_quad(methode, a, b):
    if methode == "point_milieu":
        x_mid = (a + b) / 2
        return (b - a) * f(x_mid)
    elif methode == "trapeze":
        return (b - a) * (f(a) + f(b)) / 2
    elif methode == "simpson":
        x_mid = (a + b) / 2
        return (b - a) / 6 * (f(a) + 4 * f(x_mid) + f(b))
    else:
        raise ValueError("Méthode inconnue.")

def Gauss_quad_composite(methode, a, b, N):
    h = (b - a) / N
    return sum(Gauss_quad(methode, a + i*h, a + (i+1)*h) for i in range(N))

# --- Interface Streamlit ---
st.set_page_config(page_title="Méthodes d'intégration numérique", layout="wide")
st.title("\U0001F522 Analyse complète des méthodes d'intégration numérique")

col1, col2, col3 = st.columns(3)
with col1:
    methode = st.selectbox("Choisissez la méthode", ["point_milieu", "trapeze", "simpson"])
with col2:
    N = st.slider("Nombre de subdivisions N", min_value=1, max_value=512, value=50)
with col3:
    a = st.number_input("Borne a", value=0.0)
    b = st.number_input("Borne b", value=10.0)

# --- Calculs de base ---
a_, b_ = min(a, b), max(a, b)
valeur_exacte, _ = quad(f, a_, b_)
if a > b:
    valeur_exacte = -valeur_exacte

x_vals = np.linspace(a_, b_, 1000)
max_f_double = np.max(np.abs(f_double_prime(x_vals)))
max_f_fourth = np.max(np.abs(f_fourth_prime(x_vals)))

# --- Résultats et erreurs pour chaque méthode ---
N_values = np.arange(1, 129)
data = []
for meth in ["point_milieu", "trapeze", "simpson"]:
    for n in N_values:
        try:
            approx = Gauss_quad_composite(meth, a_, b_, n)
            err_exact = abs(approx - valeur_exacte)
            if meth == "simpson":
                err_theo = abs((b_ - a_)**5 / (2880 * n**4) * max_f_fourth)
            else:
                denom = 24 if meth == "point_milieu" else 12
                err_theo = abs((b_ - a_)**3 / (denom * n**2) * max_f_double)
            rel_error = 100 * err_exact / abs(valeur_exacte)
            data.append((meth, n, approx, err_exact, err_theo, rel_error))
        except:
            data.append((meth, n, np.nan, np.nan, np.nan, np.nan))

df = pd.DataFrame(data, columns=["Méthode", "N", "Approximation", "Erreur exacte", "Erreur théorique", "Erreur relative (%)"])

# --- Résultats sélectionnés ---
df_selected = df[(df["Méthode"] == methode) & (df["N"] == N)]

if not df_selected.empty:
    approx = df_selected["Approximation"].values[0]
    err_exacte = df_selected["Erreur exacte"].values[0]
    err_theorique = df_selected["Erreur théorique"].values[0]
    rel_error = df_selected["Erreur relative (%)"].values[0]
    
    st.markdown("### 📌 Résumé des résultats")
    st.code(f"""
    Valeur exacte         : {valeur_exacte:.10f}
    Méthode               : {methode}
    Nombre de subdivisions: {N}
    Valeur approchée      : {approx:.10f}
    Erreur exacte         : {err_exacte:.5e}
    Erreur relative (%)   : {rel_error:.3f} %
    Erreur théorique      : {err_theorique:.5e}
    """, language="text")
else:
    st.warning(f"Aucune donnée disponible pour la méthode '{methode}' avec N = {N}.")


# --- Résumé ---
st.markdown("### \U0001F4CC Résumé des résultats")
st.code(f"""
Valeur exacte         : {valeur_exacte:.10f}
Méthode               : {methode}
Nombre de subdivisions: {N}
Valeur approchée      : {approx:.10f}
Erreur exacte         : {err_exacte:.5e}
Erreur relative (%)   : {rel_error:.3f} %
Erreur théorique      : {err_theorique:.5e}
""", language="text")

# --- Graphique de la fonction ---
st.markdown("### \U0001F4CA Graphique de la fonction f(x)")
fig1, ax1 = plt.subplots()
ax1.plot(x_vals, f(x_vals), label="$f(x) = e^{-x^2}$", color='blue')
ax1.set_title("Fonction f(x)")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# --- Erreurs exactes ---
st.markdown("### \U0001F4C8 Comparaison des erreurs exactes")
fig2, ax2 = plt.subplots()
for meth in df["Méthode"].unique():
    subset = df[df["Méthode"] == meth]
    ax2.plot(subset["N"], subset["Erreur exacte"], label=meth)
ax2.set_yscale("log")
ax2.set_title("Erreur exacte vs N")
ax2.set_xlabel("N")
ax2.set_ylabel("Erreur exacte")
ax2.grid(True, which="both")
ax2.legend()
st.pyplot(fig2)

# --- Erreurs théoriques ---
st.markdown("### \U0001F4C9 Comparaison des erreurs théoriques")
fig3, ax3 = plt.subplots()
for meth in df["Méthode"].unique():
    subset = df[df["Méthode"] == meth]
    ax3.plot(subset["N"], subset["Erreur théorique"], linestyle='--', label=meth)
ax3.set_yscale("log")
ax3.set_title("Erreur théorique vs N")
ax3.set_xlabel("N")
ax3.set_ylabel("Erreur théorique")
ax3.grid(True, which="both")
ax3.legend()
st.pyplot(fig3)

# --- Erreurs relatives ---
st.markdown("### \U0001F4C5 Comparaison des erreurs relatives (%)")
fig4, ax4 = plt.subplots()
for meth in df["Méthode"].unique():
    subset = df[df["Méthode"] == meth]
    ax4.plot(subset["N"], subset["Erreur relative (%)"], label=meth)
ax4.set_yscale("log")
ax4.set_title("Erreur relative (%) vs N")
ax4.set_xlabel("N")
ax4.set_ylabel("Erreur relative (%)")
ax4.grid(True, which="both")
ax4.legend()
st.pyplot(fig4)

# --- Formules utilisées ---
with st.expander("📝 Formules utilisées"):
    st.markdown(r"""
    ### Méthodes d'intégration :

    - **Point Milieu :**  
      $$ \int_a^b f(x) \, dx \approx (b - a) \cdot f\left( \frac{a + b}{2} \right) $$

    - **Trapèzes :**  
      $$ \int_a^b f(x) \, dx \approx \frac{b - a}{2} [f(a) + f(b)] $$

    - **Simpson :**  
      $$ \int_a^b f(x) \, dx \approx \frac{b - a}{6} [f(a) + 4f(m) + f(b)] $$

    ### Erreurs théoriques :

    - **Point Milieu :**  
      $$ \text{Erreur} \leq \frac{(b-a)^3}{24N^2} \cdot \max |f''(x)| $$

    - **Trapèzes :**  
      $$ \text{Erreur} \leq \frac{(b-a)^3}{12N^2} \cdot \max |f''(x)| $$

    - **Simpson :**  
      $$ \text{Erreur} \leq \frac{(b-a)^5}{2880N^4} \cdot \max |f^{(4)}(x)| $$
    """, unsafe_allow_html=True)

# --- Téléchargement des données ---
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("\U0001F4BE Télécharger les résultats (CSV)", data=csv, file_name='resultats_integration.csv', mime='text/csv')

# --- Tableau interactif ---
st.markdown("### \U0001F522 Tableau des résultats complets")
st.dataframe(df[df["N"] <= 128].pivot(index="N", columns="Méthode", values=["Approximation", "Erreur exacte", "Erreur théorique", "Erreur relative (%)"]))

st.caption("Projet complet — Intégration numérique avec analyse comparative des méthodes.")
