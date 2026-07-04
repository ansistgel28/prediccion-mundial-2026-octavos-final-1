# ============================================================
# 04_mostrar_modelo_final.py
# Muestra el modelo ganador, coeficientes o importancia de variables
# ============================================================

import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

VARIABLES_MODELO = [
    "home_gf12",
    "home_ga12",
    "home_pts12",
    "home_prev_matches",
    "away_gf12",
    "away_ga12",
    "away_pts12",
    "away_prev_matches",
    "diff_fifa",
    "diff_elo",
    "h2h",
    "neutral",
    "home_advantage",
]


def limpiar_nombre_modelo(nombre):
    return (
        nombre.lower()
        .replace(" ", "_")
        .replace("ó", "o")
        .replace("í", "i")
        .replace("á", "a")
        .replace("é", "e")
        .replace("ú", "u")
    )


def obtener_modelo_interno(modelo):
    if hasattr(modelo, "named_steps"):
        if "modelo" in modelo.named_steps:
            return modelo.named_steps["modelo"]

        nombres = list(modelo.named_steps.keys())
        return modelo.named_steps[nombres[-1]]

    return modelo


def mostrar_coeficientes(nombre_modelo, modelo, tipo):
    modelo_interno = obtener_modelo_interno(modelo)

    print()
    print("========================================")
    print("MODELO:", nombre_modelo)
    print("TIPO:", tipo)
    print("========================================")

    if hasattr(modelo_interno, "coef_"):
        intercepto = modelo_interno.intercept_
        coeficientes = modelo_interno.coef_

        try:
            intercepto = float(intercepto)
        except Exception:
            intercepto = float(intercepto[0])

        print("Intercepto:", round(intercepto, 6))
        print()

        print("Coeficientes:")
        for variable, coef in zip(VARIABLES_MODELO, coeficientes):
            print(variable, "=", round(float(coef), 6))

        print()
        print("Ecuacion del modelo:")
        print("Y =", round(intercepto, 6), end=" ")

        for variable, coef in zip(VARIABLES_MODELO, coeficientes):
            coef = float(coef)

            if coef >= 0:
                signo = "+"
            else:
                signo = "-"

            print(signo, abs(round(coef, 6)), "*", variable, end=" ")

        print()
        print("========================================")

    elif hasattr(modelo_interno, "feature_importances_"):
        importancias = pd.DataFrame({
            "variable": VARIABLES_MODELO,
            "importancia": modelo_interno.feature_importances_
        })

        importancias = importancias.sort_values(
            "importancia",
            ascending=False
        )

        print("Este modelo no tiene ecuacion lineal directa.")
        print("Se interpreta mediante importancia de variables:")
        print()
        print(importancias.to_string(index=False))
        print("========================================")

    else:
        print("No se pudieron extraer coeficientes ni importancias.")
        print("========================================")


def main():
    archivo_mejor_modelo = OUT_DIR / "mejor_modelo.txt"

    if not archivo_mejor_modelo.exists():
        raise FileNotFoundError("No existe outputs/mejor_modelo.txt")

    mejor_modelo = archivo_mejor_modelo.read_text(encoding="utf-8").strip()
    nombre_archivo = limpiar_nombre_modelo(mejor_modelo)

    print()
    print("========================================")
    print("MODELO GANADOR GLOBAL")
    print("========================================")
    print(mejor_modelo)
    print("========================================")

    modelo_home_path = OUT_DIR / f"modelo_home_{nombre_archivo}.pkl"
    modelo_away_path = OUT_DIR / f"modelo_away_{nombre_archivo}.pkl"

    if not modelo_home_path.exists():
        modelo_home_path = OUT_DIR / "modelo_home.pkl"

    if not modelo_away_path.exists():
        modelo_away_path = OUT_DIR / "modelo_away.pkl"

    if not modelo_home_path.exists():
        raise FileNotFoundError("No existe el modelo home guardado.")

    if not modelo_away_path.exists():
        raise FileNotFoundError("No existe el modelo away guardado.")

    modelo_home = joblib.load(modelo_home_path)
    modelo_away = joblib.load(modelo_away_path)

    mostrar_coeficientes(
        nombre_modelo=mejor_modelo,
        modelo=modelo_home,
        tipo="Prediccion de goles del local"
    )

    mostrar_coeficientes(
        nombre_modelo=mejor_modelo,
        modelo=modelo_away,
        tipo="Prediccion de goles del visitante"
    )


if __name__ == "__main__":
    main()