"""
Vinho Verde — Predictor de Calidad y Estudio de Relevancia del Alcohol
Streamlit app | Modelo: CatBoost | Dataset: UCI Wine Quality (red + white)
"""

import json
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Vinho Verde — Calidad del Vino",
    page_icon="🍷",
    layout="wide",
)

# ── Paleta ──────────────────────────────────────────────────────────────────
COLOR_ALCOHOL = "#C0392B"
COLOR_BG_CARD = "#F8F4F9"

# ── Carga de artefactos (cacheados) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_wine.cbm")
    return model

@st.cache_data
def load_schema():
    with open("input_schema.json", "r") as f:
        return json.load(f)

model  = load_model()
schema = load_schema()

feat_map = {f["name"]: f for f in schema["features"]}
FEATURES = [f["name"] for f in schema["features"]]

# ── Labels y metadatos por variable ─────────────────────────────────────────
LABELS_ES = {
    "fixed acidity":        "Acidez fija",
    "volatile acidity":     "Acidez volátil",
    "citric acid":          "Ácido cítrico",
    "residual sugar":       "Azúcar residual",
    "chlorides":            "Cloruros",
    "free sulfur dioxide":  "SO₂ libre",
    "total sulfur dioxide": "SO₂ total",
    "density":              "Densidad",
    "pH":                   "pH",
    "sulphates":            "Sulfatos",
    "alcohol":              "Alcohol (% vol)",
}

UNITS = {
    "fixed acidity":        "g/L",
    "volatile acidity":     "g/L",
    "citric acid":          "g/L",
    "residual sugar":       "g/L",
    "chlorides":            "g/L",
    "free sulfur dioxide":  "mg/L",
    "total sulfur dioxide": "mg/L",
    "density":              "g/cm³",
    "pH":                   "",
    "sulphates":            "g/L",
    "alcohol":              "% vol",
}

TOOLTIPS = {
    "fixed acidity":        "Ácidos que no se evaporan fácilmente. Aportan estructura al vino.",
    "volatile acidity":     "Ácido acético. En exceso produce olor a vinagre — perjudica la calidad.",
    "citric acid":          "Aporta frescura. Presente en pequeñas cantidades en vinos buenos.",
    "residual sugar":       "Azúcar que queda tras la fermentación. Varía mucho entre blanco y tinto.",
    "chlorides":            "Sales minerales. En exceso dan sabor salado.",
    "free sulfur dioxide":  "SO₂ libre que actúa como conservante y antioxidante.",
    "total sulfur dioxide": "Total de SO₂ (libre + combinado). Regulado por normas de calidad.",
    "density":              "Relacionada con el contenido de alcohol y azúcar.",
    "pH":                   "Medida de acidez global. Los vinos suelen estar entre 3.0 y 3.5.",
    "sulphates":            "Aditivo que contribuye a los niveles de SO₂ y actúa como conservante.",
    "alcohol":              "Contenido alcohólico. La variable más relevante para predecir calidad.",
}

STEPS = {
    "density":          0.0001,
    "chlorides":        0.001,
    "volatile acidity": 0.01,
    "citric acid":      0.01,
    "sulphates":        0.01,
    "pH":               0.01,
    "fixed acidity":    0.1,
    "alcohol":          0.1,
    "residual sugar":   0.1,
    "free sulfur dioxide":  1.0,
    "total sulfur dioxide": 1.0,
}

def quality_label(q: float) -> str:
    if q >= 7.5: return "Excelente 🌟"
    if q >= 6.5: return "Muy bueno 👍"
    if q >= 5.5: return "Bueno ✅"
    if q >= 4.5: return "Regular 😐"
    return "Deficiente ⚠️"

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — ENCABEZADO E INTRODUCCIÓN
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='color:{COLOR_ALCOHOL}; margin-bottom:0'>🍷 Vinho Verde — Predictor de Calidad</h1>"
    "<p style='color:#555; font-size:1.05rem; margin-top:4px'>"
    "Dataset UCI Wine Quality · Red + White · 6.492 muestras · Modelo CatBoost</p>",
    unsafe_allow_html=True,
)

with st.expander("📖 ¿De qué trata esto? (léeme si es tu primera vez)", expanded=False):
    st.markdown("""
    ### El experimento

    El **Vinho Verde** es un vino portugués producido en la región del Minho.
    Este estudio analiza **6.492 muestras** de vinos tintos y blancos, cada una caracterizada
    por 11 propiedades fisicoquímicas medidas en laboratorio, como el nivel de alcohol,
    la acidez o los sulfatos.

    Para cada muestra, un panel de catadores entrenados le asignó una **nota de calidad entre 0 y 10**.

    ### La pregunta

    ¿Podemos predecir la calidad de un vino **solo mirando sus propiedades químicas**?
    Y más importante aún: **¿cuál de esas propiedades importa más?**

    ### El hallazgo principal

    Entrenamos un modelo de Machine Learning (CatBoost) y aplicamos tres métodos de
    interpretabilidad independientes. Los tres coinciden en lo mismo:

    > **El contenido de alcohol es la variable más relevante para predecir la calidad del vino.**

    Esto no significa que un vino de alta graduación sea automáticamente bueno,
    pero sí que el alcohol es la señal fisicoquímica más informativa que captan
    los catadores, consciente o inconscientemente.

    ### Cómo usar esta app

    Mueve los sliders de la sección **Predictor** para simular las propiedades de un vino
    y ve cómo cambia la nota estimada. Luego revisa la sección **¿Por qué el alcohol?**
    para ver la evidencia detrás del hallazgo.

    ---
    **Fuente:** Cortez et al. (2009) — *Modeling wine preferences by data mining from physicochemical properties.*
    Decision Support Systems, Elsevier, 47(4):547–553.
    """)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔬 Predictor de calidad")
st.caption("Ajusta las propiedades fisicoquímicas y obtén una nota estimada. "
           "Los valores por defecto corresponden a la mediana del dataset. "
           "Los rangos cubren el 90% central de los datos observados (p5–p95).")

col_tipo, _ = st.columns([1, 3])
with col_tipo:
    tipo = st.selectbox(
        "Tipo de vino",
        options=["red", "white"],
        format_func=lambda x: "🔴 Tinto (Red)" if x == "red" else "⚪ Blanco (White)",
    )

st.markdown("#### Propiedades fisicoquímicas")

slider_vals = {}
left_col, right_col = st.columns(2)

for i, feat in enumerate(FEATURES):
    info   = feat_map[feat]
    label  = LABELS_ES.get(feat, feat)
    unit   = UNITS.get(feat, "")
    tip    = TOOLTIPS.get(feat, "")
    step   = STEPS.get(feat, 0.1)
    p05    = float(info["p05"])
    p95    = float(info["p95"])
    median = float(info["median"])

    col = left_col if i % 2 == 0 else right_col

    with col:
        if feat == "alcohol":
            st.markdown(
                f"<span style='color:{COLOR_ALCOHOL}; font-weight:700'>"
                f"⭐ {label} — variable clave</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{label}** {'(' + unit + ')' if unit else ''}")

        st.caption(tip)

        slider_vals[feat] = st.slider(
            label=label,
            min_value=p05,
            max_value=p95,
            value=median,
            step=step,
            label_visibility="collapsed",
            key=f"slider_{feat}",
        )

st.markdown("")
btn_col, res_col = st.columns([1, 3])

with btn_col:
    predecir = st.button("🍷 Predecir calidad", type="primary", use_container_width=True)
    if st.button("↺ Restaurar medianas", use_container_width=True):
        for feat in FEATURES:
            key = f"slider_{feat}"
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if predecir:
    row             = {**slider_vals, "tipo": tipo}
    input_df        = pd.DataFrame([row], columns=FEATURES + ["tipo"])
    pred            = float(model.predict(input_df)[0])
    pred            = round(max(3.0, min(8.0, pred)), 2)
    label_pred      = quality_label(pred)
    alc             = slider_vals["alcohol"]

    with res_col:
        st.markdown(
            f"<div style='background:{COLOR_BG_CARD}; border-left:5px solid {COLOR_ALCOHOL};"
            f"border-radius:12px; padding:20px; margin-top:8px'>"
            f"<h2 style='margin:0; color:{COLOR_ALCOHOL}'>Nota estimada: {pred} / 10</h2>"
            f"<h3 style='margin:4px 0 12px 0; color:#333'>{label_pred}</h3>"
            f"<p style='margin:0; color:#555; font-size:0.95rem'>"
            f"Tipo: <b>{'Tinto' if tipo == 'red' else 'Blanco'}</b> · "
            f"Alcohol: <b>{alc}% vol</b></p></div>",
            unsafe_allow_html=True,
        )
        if alc >= 11.5:
            st.info("💡 El nivel de alcohol está en el rango típico de los vinos bien valorados (≥ 11.5% vol).")
        elif alc <= 9.5:
            st.warning("💡 El nivel de alcohol es bajo — según el modelo, esto tiende a reducir la nota estimada.")
        else:
            st.info("💡 Nivel de alcohol medio. Prueba subirlo para ver cómo cambia la predicción.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ¿POR QUÉ EL ALCOHOL?
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 ¿Por qué el alcohol es la variable más relevante?")
st.markdown(
    "Aplicamos **tres métodos de interpretabilidad independientes**. "
    "Que los tres lleguen al mismo resultado hace el hallazgo sólido ante cualquier cuestionamiento."
)

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Feature Importance",
    "2️⃣ Permutation Importance",
    "3️⃣ SHAP",
])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image("plots/feature_importance.png", use_container_width=True)
    with c2:
        st.markdown("### ¿Qué mide?")
        st.markdown("""
        Mide cuánto contribuye cada variable a reducir el error del modelo
        durante el entrenamiento — internamente, a través de los splits de los árboles.

        **Resultado:** El alcohol lidera con ~13%, más que cualquier otra variable.

        ⚠️ *Al medir el uso interno del modelo puede tener sesgos.
        Por eso se complementa con los otros dos métodos.*
        """)

with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image("plots/permutation_importance.png", use_container_width=True)
    with c2:
        st.markdown("### ¿Qué mide?")
        st.markdown("""
        Toma cada variable y mezcla sus valores aleatoriamente — rompiendo
        su relación con la calidad. Luego mide cuánto empeoró el modelo.

        Si mezclar el alcohol daña mucho el modelo, es porque era crucial.

        **Resultado:** Permutar el alcohol produce el mayor daño (~0.175),
        casi el doble que la segunda variable.
        Las barras de error pequeñas confirman que el resultado es **estable**.
        """)

with tab3:
    c1, c2 = st.columns([2, 1])
    with c1:
        t3a, t3b = st.tabs(["Vista general", "Efecto del alcohol"])
        with t3a:
            st.image("plots/shap_beeswarm.png", use_container_width=True)
        with t3b:
            st.image("plots/shap_dependence_alcohol.png", use_container_width=True)
    with c2:
        st.markdown("### ¿Qué mide?")
        st.markdown("""
        SHAP calcula, para **cada vino individual**, cuánto empujó cada variable
        la predicción hacia arriba o hacia abajo respecto al promedio.

        **Vista general (Beeswarm):** Cada punto es un vino.
        Los puntos rojos (alcohol alto) se concentran a la derecha (+),
        los azules (alcohol bajo) a la izquierda (−).
        La separación es la más clara de todas las variables.

        **Efecto del alcohol:** A partir de ~11% vol el alcohol
        empieza a sumar nota consistentemente, tanto en tintos como en blancos.
        """)

st.divider()

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.88rem; padding:12px 0'>"
    "<b>Dataset:</b> Cortez et al. (2009) · UCI Wine Quality · "
    "<b>Modelo:</b> CatBoost + Optuna · "
    "<b>Interpretabilidad:</b> Feature Importance + Permutation Importance + SHAP<br>"
    "Universidad Bernardo O'Higgins · Facultad de Ingeniería y Negocios"
    "</div>",
    unsafe_allow_html=True,
)
