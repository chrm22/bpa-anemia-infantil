"""
App de predicción de riesgo de anemia infantil — ENDES 2024
Grupo 2 · Business Predictive Analytics (1ASI0709)

Carga el artefacto 'modelo_final_anemia.pkl' (preprocesador + Random Forest +
umbral ajustado) y predice el riesgo de anemia de un niño a partir de datos
crudos ingresados en un formulario. Las variables derivadas
(ratio_peso_talla, ratio_peso_edad, estado_nutricional) se recrean aquí con la
misma lógica del notebook, antes de pasar los datos al preprocesador.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN
# --------------------------------------------------------------------------- #
RUTA_MODELO = "modelo_final_anemia.pkl"

# En la corrida final, RFECV no eliminó ninguna variable cruda por completo
# (solo descartó 2 categorías minoritarias dentro de variables que conservaron
# el resto). Por eso el formulario pide todas las variables originales.
VARIABLES_NOMINALES_DESCARTADAS = []

# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Predicción de anemia infantil", page_icon="🩸",
                   layout="centered")


@st.cache_resource
def cargar_artefacto(ruta):
    return joblib.load(ruta)


# --------------------------------------------------------------------------- #
# Feature engineering — misma lógica que el notebook
# --------------------------------------------------------------------------- #
def construir_features(entrada: dict, umbrales: dict) -> pd.DataFrame:
    X = pd.DataFrame([entrada])

    X["ratio_peso_talla"] = X["peso_kg"] / X["talla_cm"]
    edad_no_cero = X["edad_meses"].replace(0, np.nan)
    X["ratio_peso_edad"] = X["peso_kg"] / edad_no_cero

    edad = int(X["edad_meses"].iloc[0])
    ref_peso = umbrales["peso"].get(edad, umbrales["peso_median"])
    ref_talla = umbrales["talla"].get(edad, umbrales["talla_median"])
    bajo_peso = int(X["peso_kg"].iloc[0] < ref_peso)
    baja_talla = int(X["talla_cm"].iloc[0] < ref_talla)
    X["estado_nutricional"] = bajo_peso + baja_talla

    return X


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #
try:
    art = cargar_artefacto(RUTA_MODELO)
except FileNotFoundError:
    st.error(f"No se encontró '{RUTA_MODELO}'. Debe estar en la raíz del repositorio.")
    st.stop()

preprocessor = art["preprocessor"]
modelo = art["modelo"]
umbral = art["umbral"]
features_entrada = art["features_entrada"]
umbrales = art["umbrales_nutricion"]
umbrales["peso"] = {int(k): v for k, v in umbrales["peso"].items()}
umbrales["talla"] = {int(k): v for k, v in umbrales["talla"].items()}


# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.title("🩸 Predicción de riesgo de anemia infantil")
st.caption(
    "Herramienta de apoyo a la focalización del tamizaje — ENDES 2024. "
    "No reemplaza el diagnóstico clínico de hemoglobina."
)
st.markdown("Complete los datos del niño, de la madre y del hogar:")

# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #
with st.form("formulario_nino"):

    st.subheader("👶 Datos del niño")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad_meses = st.number_input("Edad (meses)", 0, 59, 24)
        sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
    with c2:
        peso_kg = st.number_input("Peso (kg)", 2.0, 30.0, 12.0, step=0.1)
        orden_nacimiento = st.number_input("Orden de nacimiento", 1, 15, 1)
    with c3:
        talla_cm = st.number_input("Talla (cm)", 40.0, 130.0, 85.0, step=0.1)
        intervalo_nacimiento = st.number_input(
            "Intervalo desde nacimiento anterior (meses, 0 si es primogénito)",
            0, 300, 0)

    st.subheader("🤱 Datos de la madre")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad_madre = st.number_input("Edad de la madre", 12, 60, 28)
        educ_madre = st.selectbox("Educación de la madre",
                                  ["Sin educación", "Primaria", "Secundaria", "Superior"])
    with c2:
        imc_madre = st.number_input("IMC de la madre", 12.0, 50.0, 25.0, step=0.1)
        madre_embarazada = st.selectbox("¿Madre embarazada?", ["No", "Sí"])
    with c3:
        anemia_madre = st.selectbox("¿Madre con anemia?", ["No", "Sí"])

    st.subheader("🏠 Datos del hogar")
    c1, c2, c3 = st.columns(3)
    with c1:
        region = st.number_input("Región (código INEI)", 1, 25, 15)
        area = st.selectbox("Área", ["Urbano", "Rural"])
        altitud = st.number_input("Altitud (msnm)", 0, 5500, 200)
    with c2:
        indice_riqueza = st.selectbox(
            "Índice de riqueza",
            ["Más pobre", "Pobre", "Medio", "Rico", "Más rico"])
        total_personas_hogar = st.number_input("Total de personas en el hogar", 1, 20, 4)
        ninos_menores5_hogar = st.number_input("Niños menores de 5 años", 0, 10, 1)
    with c3:
        habitaciones_dormir = st.number_input("Habitaciones para dormir", 1, 15, 2)
        electricidad = st.selectbox("¿Tiene electricidad?", ["Sí", "No"])
        fuente_agua_cat = st.selectbox(
            "Fuente de agua", ["Red publica", "Pozo", "Embotellada", "Otra fuente"])

    c1, c2 = st.columns(2)
    with c1:
        sshh_cat = st.selectbox("Tipo de servicio higiénico",
                                ["Mejorado", "Basico", "Sin servicio"])
        piso_cat = st.selectbox("Material del piso",
                                ["Acabado", "Madera", "Tierra", "Otro"])
    with c2:
        combustible_cat = st.selectbox("Combustible para cocinar",
                                       ["Limpio", "Contaminante", "No cocina"])

    enviar = st.form_submit_button("Predecir riesgo", use_container_width=True)


# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #
if enviar:
    entrada = {
        "edad_meses": edad_meses,
        "peso_kg": peso_kg,
        "talla_cm": talla_cm,
        "sexo": 1 if sexo == "Hombre" else 2,
        "orden_nacimiento": orden_nacimiento,
        "intervalo_nacimiento": intervalo_nacimiento,
        "edad_madre": edad_madre,
        "educ_madre": educ_madre,
        "imc_madre": imc_madre,
        "madre_embarazada": 1 if madre_embarazada == "Sí" else 0,
        "anemia_madre": 1 if anemia_madre == "Sí" else 0,
        "region": region,
        "area": 1 if area == "Urbano" else 2,
        "altitud": altitud,
        "indice_riqueza": indice_riqueza,
        "total_personas_hogar": total_personas_hogar,
        "ninos_menores5_hogar": ninos_menores5_hogar,
        "habitaciones_dormir": habitaciones_dormir,
        "electricidad": 1 if electricidad == "Sí" else 0,
        "fuente_agua_cat": fuente_agua_cat,
        "sshh_cat": sshh_cat,
        "piso_cat": piso_cat,
        "combustible_cat": combustible_cat,
    }

    X = construir_features(entrada, umbrales)

    faltantes = [c for c in features_entrada if c not in X.columns]
    if faltantes:
        st.error(f"Faltan columnas que el modelo espera: {faltantes}.")
        st.stop()
    X = X[features_entrada]

    X_t = preprocessor.transform(X)
    proba = float(modelo.predict_proba(X_t)[:, 1][0])
    pred = int(proba >= umbral)

    st.divider()
    st.subheader("Resultado")

    col_a, col_b = st.columns(2)
    col_a.metric("Probabilidad de anemia", f"{proba*100:.1f}%")
    col_b.metric("Umbral de decisión", f"{umbral*100:.1f}%")

    if pred == 1:
        st.error("⚠️ **Riesgo ALTO de anemia** — se recomienda priorizar el tamizaje de hemoglobina.")
    else:
        st.success("✅ **Riesgo BAJO de anemia** — seguimiento según protocolo regular.")

    st.progress(min(proba, 1.0))
    st.caption(
        f"Estado nutricional estimado: {int(X['estado_nutricional'].iloc[0])} "
        "(0 = peso y talla normales · 1 = uno bajo · 2 = ambos bajos)"
    )

    with st.expander("Ver detalle técnico"):
        st.write("Variables derivadas calculadas:")
        st.dataframe(X[["ratio_peso_talla", "ratio_peso_edad", "estado_nutricional"]])
        st.write(f"Variables usadas por el modelo ({len(features_entrada)}):")
        st.code(", ".join(features_entrada))

    st.info(
        "Esta predicción es una **estimación de riesgo** basada en variables "
        "socioeconómicas y demográficas, no un diagnóstico. Debe usarse bajo "
        "supervisión de personal de salud."
    )
