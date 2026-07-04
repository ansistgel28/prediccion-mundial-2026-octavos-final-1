# ============================================================
# 07_predecir_octavos.py
# Predice quienes pasan de 16avos a octavos
# Mundial 2026
# ============================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_FIXTURES_MODELO = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_FIXTURES_16AVOS = DATA_DIR / "fixtures_16avos.csv"

ARCHIVO_MODELO_HOME = OUT_DIR / "modelo_home.pkl"
ARCHIVO_MODELO_AWAY = OUT_DIR / "modelo_away.pkl"

ARCHIVO_SALIDA_16AVOS = OUT_DIR / "prediccion_16avos_a_octavos.csv"
ARCHIVO_OCTAVOS = OUT_DIR / "clasificados_octavos.csv"
IMAGEN_16AVOS = OUT_DIR / "panel_16avos_a_octavos.png"

MAX_GOLES = 8

VARIABLES = [
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


# ============================================================
# CARGA DE DATOS
# ============================================================

def validar_archivos():
    if not ARCHIVO_FIXTURES_MODELO.exists():
        raise FileNotFoundError("No existe data/fixtures_modelo.csv")

    if not ARCHIVO_FIXTURES_16AVOS.exists():
        raise FileNotFoundError(
            "No existe data/fixtures_16avos.csv. "
            "Debes crear este archivo con los partidos actuales de 16avos."
        )

    if not ARCHIVO_MODELO_HOME.exists():
        raise FileNotFoundError("No existe outputs/modelo_home.pkl. Ejecuta primero el 02.")

    if not ARCHIVO_MODELO_AWAY.exists():
        raise FileNotFoundError("No existe outputs/modelo_away.pkl. Ejecuta primero el 02.")

    OUT_DIR.mkdir(exist_ok=True)


def cargar_datos():
    validar_archivos()

    fixtures_modelo = pd.read_csv(ARCHIVO_FIXTURES_MODELO)
    fixtures_16avos = pd.read_csv(ARCHIVO_FIXTURES_16AVOS)

    fixtures_modelo.columns = fixtures_modelo.columns.str.strip()
    fixtures_16avos.columns = fixtures_16avos.columns.str.strip()

    if "match_id" not in fixtures_modelo.columns and "match_no" in fixtures_modelo.columns:
        fixtures_modelo = fixtures_modelo.rename(columns={"match_no": "match_id"})

    columnas_16avos = [
        "match_id",
        "date",
        "home_team",
        "away_team"
    ]

    for col in columnas_16avos:
        if col not in fixtures_16avos.columns:
            raise ValueError("Falta columna en fixtures_16avos.csv: " + col)

    fixtures_16avos["date"] = pd.to_datetime(
        fixtures_16avos["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    fixtures_16avos["match_id"] = pd.to_numeric(
        fixtures_16avos["match_id"],
        errors="coerce"
    ).astype(int)

    if "round" not in fixtures_16avos.columns:
        fixtures_16avos["round"] = "16AVOS"

    if "home_score" not in fixtures_16avos.columns:
        fixtures_16avos["home_score"] = pd.NA

    if "away_score" not in fixtures_16avos.columns:
        fixtures_16avos["away_score"] = pd.NA

    if "pen_home" not in fixtures_16avos.columns:
        fixtures_16avos["pen_home"] = pd.NA

    if "pen_away" not in fixtures_16avos.columns:
        fixtures_16avos["pen_away"] = pd.NA

    modelo_home = joblib.load(ARCHIVO_MODELO_HOME)
    modelo_away = joblib.load(ARCHIVO_MODELO_AWAY)

    return fixtures_modelo, fixtures_16avos, modelo_home, modelo_away


# ============================================================
# PERFIL DE EQUIPOS DESDE fixtures_modelo.csv
# ============================================================

def construir_perfiles_equipos(fixtures_modelo):
    filas = []

    for _, fila in fixtures_modelo.iterrows():
        if "home_team" in fila.index:
            filas.append({
                "team": fila["home_team"],
                "gf12": fila.get("home_gf12", 0),
                "ga12": fila.get("home_ga12", 0),
                "pts12": fila.get("home_pts12", 0),
                "prev_matches": fila.get("home_prev_matches", 0),
                "fifa": fila.get("fifa_home", 0),
                "elo": fila.get("elo_home", 0),
            })

        if "away_team" in fila.index:
            filas.append({
                "team": fila["away_team"],
                "gf12": fila.get("away_gf12", 0),
                "ga12": fila.get("away_ga12", 0),
                "pts12": fila.get("away_pts12", 0),
                "prev_matches": fila.get("away_prev_matches", 0),
                "fifa": fila.get("fifa_away", 0),
                "elo": fila.get("elo_away", 0),
            })

    perfiles = pd.DataFrame(filas)

    for col in ["gf12", "ga12", "pts12", "prev_matches", "fifa", "elo"]:
        perfiles[col] = pd.to_numeric(
            perfiles[col],
            errors="coerce"
        )

    perfiles = perfiles.groupby("team", as_index=False).mean(numeric_only=True)
    perfiles = perfiles.fillna(0)

    return perfiles


def obtener_perfil(perfiles, equipo):
    fila = perfiles[perfiles["team"] == equipo]

    if fila.empty:
        raise ValueError(
            "No se encontró perfil para el equipo: " + str(equipo)
            + ". Revisa que el nombre coincida con fixtures_modelo.csv."
        )

    return fila.iloc[0]


def construir_features_partido(perfiles, home_team, away_team):
    home = obtener_perfil(perfiles, home_team)
    away = obtener_perfil(perfiles, away_team)

    datos = {
        "home_gf12": float(home["gf12"]),
        "home_ga12": float(home["ga12"]),
        "home_pts12": float(home["pts12"]),
        "home_prev_matches": float(home["prev_matches"]),
        "away_gf12": float(away["gf12"]),
        "away_ga12": float(away["ga12"]),
        "away_pts12": float(away["pts12"]),
        "away_prev_matches": float(away["prev_matches"]),
        "diff_fifa": float(home["fifa"]) - float(away["fifa"]),
        "diff_elo": float(home["elo"]) - float(away["elo"]),
        "h2h": 0,
        "neutral": 1,
        "home_advantage": 0,
    }

    return pd.DataFrame([datos], columns=VARIABLES)


# ============================================================
# MATRIZ POISSON
# ============================================================

def crear_matriz_probabilidades(lambda_home, lambda_away):
    matriz = []

    for goles_home in range(MAX_GOLES + 1):
        fila = []

        for goles_away in range(MAX_GOLES + 1):
            prob = poisson.pmf(goles_home, lambda_home) * poisson.pmf(goles_away, lambda_away)
            fila.append(prob)

        matriz.append(fila)

    matriz = pd.DataFrame(matriz)

    suma_total = matriz.values.sum()

    if suma_total > 0:
        matriz = matriz / suma_total

    return matriz


def calcular_probabilidades(matriz):
    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = float(matriz.iloc[i, j])

            if i > j:
                prob_home += valor
            elif i == j:
                prob_draw += valor
            else:
                prob_away += valor

    return prob_home, prob_draw, prob_away


def obtener_marcador_mas_probable(matriz):
    mejor_prob = -1
    mejor_home = 0
    mejor_away = 0

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            prob = matriz.iloc[i, j]

            if prob > mejor_prob:
                mejor_prob = prob
                mejor_home = i
                mejor_away = j

    return mejor_home, mejor_away, mejor_prob


# ============================================================
# GANADOR REAL SI YA EXISTE RESULTADO
# ============================================================

def obtener_ganador_real(fila):
    home_score = pd.to_numeric(fila.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(fila.get("away_score"), errors="coerce")
    pen_home = pd.to_numeric(fila.get("pen_home"), errors="coerce")
    pen_away = pd.to_numeric(fila.get("pen_away"), errors="coerce")

    if pd.isna(home_score) or pd.isna(away_score):
        return None

    home_team = fila["home_team"]
    away_team = fila["away_team"]

    home_score = int(home_score)
    away_score = int(away_score)

    if home_score > away_score:
        return {
            "winner": home_team,
            "source": "REAL_90_120",
            "score": str(home_score) + "-" + str(away_score)
        }

    if away_score > home_score:
        return {
            "winner": away_team,
            "source": "REAL_90_120",
            "score": str(home_score) + "-" + str(away_score)
        }

    if not pd.isna(pen_home) and not pd.isna(pen_away):
        pen_home = int(pen_home)
        pen_away = int(pen_away)

        if pen_home > pen_away:
            return {
                "winner": home_team,
                "source": "REAL_PENALES",
                "score": str(home_score) + "-" + str(away_score)
                + " pen " + str(pen_home) + "-" + str(pen_away)
            }

        if pen_away > pen_home:
            return {
                "winner": away_team,
                "source": "REAL_PENALES",
                "score": str(home_score) + "-" + str(away_score)
                + " pen " + str(pen_home) + "-" + str(pen_away)
            }

    return None


# ============================================================
# PREDICCION DE 16AVOS
# ============================================================

def predecir_partido(modelo_home, modelo_away, perfiles, fila):
    home_team = str(fila["home_team"])
    away_team = str(fila["away_team"])

    resultado_real = obtener_ganador_real(fila)

    X = construir_features_partido(
        perfiles=perfiles,
        home_team=home_team,
        away_team=away_team
    )

    lambda_home = float(modelo_home.predict(X)[0])
    lambda_away = float(modelo_away.predict(X)[0])

    lambda_home = max(lambda_home, 0.05)
    lambda_away = max(lambda_away, 0.05)

    matriz = crear_matriz_probabilidades(
        lambda_home=lambda_home,
        lambda_away=lambda_away
    )

    prob_home, prob_draw, prob_away = calcular_probabilidades(matriz)

    marcador_home, marcador_away, _prob_score = obtener_marcador_mas_probable(matriz)
    marcador_home = int(marcador_home)
    marcador_away = int(marcador_away)

    prob_home = float(prob_home)
    prob_draw = float(prob_draw)
    prob_away = float(prob_away)
    prob_adv_home = float(prob_home + 0.5 * prob_draw)
    prob_adv_away = float(prob_away + 0.5 * prob_draw)

    if prob_adv_home >= prob_adv_away:
        predicted_winner = home_team
    else:
        predicted_winner = away_team

    if resultado_real is not None:
        final_winner = str(resultado_real["winner"])
        source = str(resultado_real["source"])
        final_score = str(resultado_real["score"])
        acierto = int(predicted_winner == final_winner)
    else:
        final_winner = predicted_winner
        source = "PREDICHO_MODELO"
        final_score = str(marcador_home) + "-" + str(marcador_away)
        acierto = pd.NA

    return {
        "match_id": int(fila["match_id"]),
        "date": fila["date"],
        "round": fila.get("round", "16AVOS"),
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": round(lambda_home, 4),
        "lambda_away": round(lambda_away, 4),
        "prob_home_win_percent": round(prob_home * 100, 2),
        "prob_draw_percent": round(prob_draw * 100, 2),
        "prob_away_win_percent": round(prob_away * 100, 2),
        "prob_adv_home_percent": round(prob_adv_home * 100, 2),
        "prob_adv_away_percent": round(prob_adv_away * 100, 2),
        "predicted_score": str(marcador_home) + "-" + str(marcador_away),
        "predicted_winner": predicted_winner,
        "final_score_used": final_score,
        "final_winner": final_winner,
        "source": source,
        "modelo_acerto_si_hay_resultado": acierto
    }


def predecir_16avos(fixtures_16avos, modelo_home, modelo_away, perfiles):
    filas = []

    fixtures_16avos = fixtures_16avos.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    for _, fila in fixtures_16avos.iterrows():
        resultado = predecir_partido(
            modelo_home=modelo_home,
            modelo_away=modelo_away,
            perfiles=perfiles,
            fila=fila
        )

        filas.append(resultado)

    return pd.DataFrame(filas)


def construir_octavos(predicciones_16avos):
    octavos = predicciones_16avos[[
        "match_id",
        "date",
        "home_team",
        "away_team",
        "final_winner",
        "source"
    ]].copy()

    octavos = octavos.rename(columns={
        "match_id": "match_id_16avos",
        "final_winner": "team"
    })

    octavos["classification_type"] = "CLASIFICADO_A_OCTAVOS"

    return octavos


# ============================================================
# GRAFICO
# ============================================================

def graficar_16avos(predicciones_16avos):
    tabla = predicciones_16avos[[
        "match_id",
        "home_team",
        "away_team",
        "predicted_score",
        "predicted_winner",
        "final_winner",
        "source"
    ]].copy()

    tabla.columns = [
        "ID",
        "Equipo 1",
        "Equipo 2",
        "Marcador modelo",
        "Ganador modelo",
        "Pasa a octavos",
        "Fuente"
    ]

    fig, ax = plt.subplots(figsize=(20, 12), constrained_layout=False)
    ax.axis("off")

    fig.suptitle(
        "Mundial 2026 - Predicción de 16avos a octavos",
        fontsize=20,
        fontweight="bold"
    )

    tabla_plot = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center"
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(9)
    tabla_plot.scale(1, 1.5)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.90
    )

    fig.savefig(
        IMAGEN_16AVOS,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen guardada:", IMAGEN_16AVOS)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    fixtures_modelo, fixtures_16avos, modelo_home, modelo_away = cargar_datos()

    perfiles = construir_perfiles_equipos(
        fixtures_modelo=fixtures_modelo
    )

    predicciones_16avos = predecir_16avos(
        fixtures_16avos=fixtures_16avos,
        modelo_home=modelo_home,
        modelo_away=modelo_away,
        perfiles=perfiles
    )

    octavos = construir_octavos(
        predicciones_16avos=predicciones_16avos
    )

    predicciones_16avos.to_csv(
        ARCHIVO_SALIDA_16AVOS,
        index=False,
        encoding="utf-8"
    )

    octavos.to_csv(
        ARCHIVO_OCTAVOS,
        index=False,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("PREDICCION 16AVOS A OCTAVOS")
    print("========================================")
    print(predicciones_16avos.to_string(index=False))

    print()
    print("========================================")
    print("CLASIFICADOS A OCTAVOS")
    print("========================================")
    print(octavos[[
        "match_id_16avos",
        "team",
        "classification_type",
        "source"
    ]].to_string(index=False))

    evaluados = predicciones_16avos.dropna(
        subset=["modelo_acerto_si_hay_resultado"]
    ).copy()

    if len(evaluados) > 0:
        accuracy = evaluados["modelo_acerto_si_hay_resultado"].mean()

        print()
        print("========================================")
        print("VERIFICACION DEL MODELO EN PARTIDOS YA JUGADOS")
        print("========================================")
        print("Partidos evaluados:", len(evaluados))
        print("Aciertos:", int(evaluados["modelo_acerto_si_hay_resultado"].sum()))
        print("Accuracy:", round(accuracy * 100, 2), "%")

    graficar_16avos(predicciones_16avos)

    print()
    print("========================================")
    print("ARCHIVOS GENERADOS")
    print("========================================")
    print(ARCHIVO_SALIDA_16AVOS)
    print(ARCHIVO_OCTAVOS)
    print(IMAGEN_16AVOS)
    print("========================================")


if __name__ == "__main__":
    main()
