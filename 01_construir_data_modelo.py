import pandas as pd
import numpy as np
from pathlib import Path


# CONFIGURACION

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_FIFA = DATA_DIR / "fifa_ranking.csv"
ARCHIVO_ELO = DATA_DIR / "elo_ranking.csv"
ARCHIVO_FIXTURES_GRUPOS = DATA_DIR / "fixtures_2026_group_stage.csv"
ARCHIVO_FIXTURES_COMPLETO = DATA_DIR / "fixtures_2026.csv"

SALIDA_DATA_MODELO = DATA_DIR / "data_modelo.csv"
SALIDA_FIXTURES_MODELO = DATA_DIR / "fixtures_modelo.csv"

FECHA_INICIO_ENTRENAMIENTO = "2018-01-01"

# Si quieres fecha manual, coloca por ejemplo: "2026-06-28"
# Si lo dejas en None, el script toma automáticamente la última fecha con marcador + 1 día.
#FECHA_CORTE_ENTRENAMIENTO = "2026-06-22"
FECHA_CORTE_ENTRENAMIENTO = None

USAR_SOLO_OFICIALES = True
N_ULTIMOS_PARTIDOS = 12
ALPHA_TIEMPO = 0.85


# NORMALIZACION DE NOMBRES

def normalizar_nombre_equipo(nombre):
    if pd.isna(nombre):
        return nombre

    nombre = str(nombre).strip()

    mapa = {
        "USA": "United States",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "Republic of Korea": "South Korea",
        "IR Iran": "Iran",
        "Iran (Islamic Republic of)": "Iran",
        "Türkiye": "Turkey",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Czechia": "Czech Republic",
        "Congo DR": "DR Congo",
        "Democratic Republic of Congo": "DR Congo",
        "Cabo Verde": "Cape Verde",
        "Curaçao": "Curacao",
        "China PR": "China",
        "Kyrgyz Republic": "Kyrgyzstan",
        "The Gambia": "Gambia",
        "DPR Korea": "North Korea",
        "St. Kitts and Nevis": "Saint Kitts and Nevis",
        "St. Lucia": "Saint Lucia",
        "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "US Virgin Islands": "United States Virgin Islands",
    }

    return mapa.get(nombre, nombre)


def convertir_booleano(valor):
    if pd.isna(valor):
        return 0

    texto = str(valor).strip().lower()

    return int(texto in ["true", "1", "yes", "si", "sí"])


# CARGA DE DATOS

def cargar_datos():
    if not ARCHIVO_RESULTS.exists():
        raise FileNotFoundError(f"No existe {ARCHIVO_RESULTS}")

    if not ARCHIVO_FIFA.exists():
        raise FileNotFoundError(f"No existe {ARCHIVO_FIFA}")

    if not ARCHIVO_ELO.exists():
        raise FileNotFoundError(f"No existe {ARCHIVO_ELO}")

    resultados = pd.read_csv(ARCHIVO_RESULTS)
    fifa = pd.read_csv(ARCHIVO_FIFA)
    elo = pd.read_csv(ARCHIVO_ELO)

    resultados["date"] = pd.to_datetime(resultados["date"], errors="coerce")
    resultados["home_score"] = pd.to_numeric(resultados["home_score"], errors="coerce")
    resultados["away_score"] = pd.to_numeric(resultados["away_score"], errors="coerce")

    resultados = resultados.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score"
        ]
    )

    resultados["home_team"] = resultados["home_team"].apply(normalizar_nombre_equipo)
    resultados["away_team"] = resultados["away_team"].apply(normalizar_nombre_equipo)

    resultados["home_score"] = resultados["home_score"].astype(int)
    resultados["away_score"] = resultados["away_score"].astype(int)

    if "neutral" in resultados.columns:
        resultados["neutral"] = resultados["neutral"].apply(convertir_booleano)
    else:
        resultados["neutral"] = 0

    if "tournament" not in resultados.columns:
        resultados["tournament"] = "Unknown"

    if "team" not in fifa.columns or "fifa_points" not in fifa.columns:
        raise ValueError("fifa_ranking.csv debe tener columnas: team, fifa_points")

    if "team" not in elo.columns or "elo_points" not in elo.columns:
        raise ValueError("elo_ranking.csv debe tener columnas: team, elo_points")

    fifa["team"] = fifa["team"].apply(normalizar_nombre_equipo)
    elo["team"] = elo["team"].apply(normalizar_nombre_equipo)

    fifa["fifa_points"] = pd.to_numeric(fifa["fifa_points"], errors="coerce")
    elo["elo_points"] = pd.to_numeric(elo["elo_points"], errors="coerce")

    fifa = fifa.dropna(subset=["team", "fifa_points"])
    elo = elo.dropna(subset=["team", "elo_points"])

    resultados = resultados.sort_values("date").reset_index(drop=True)

    return resultados, fifa, elo


# FECHA DE CORTE AUTOMATICA

def obtener_fecha_corte_entrenamiento(resultados):
    if FECHA_CORTE_ENTRENAMIENTO is not None:
        return pd.to_datetime(FECHA_CORTE_ENTRENAMIENTO)

    fecha_maxima = resultados["date"].max()
    fecha_corte = fecha_maxima + pd.Timedelta(days=1)

    return fecha_corte


# FILTRO DE PARTIDOS

def filtrar_partidos_oficiales(resultados):
    df = resultados.copy()

    if USAR_SOLO_OFICIALES and "tournament" in df.columns:
        df = df[df["tournament"].astype(str).str.lower() != "friendly"].copy()

    return df.sort_values("date").reset_index(drop=True)

# ULTIMOS PARTIDOS Y FORMA PONDERADA
def obtener_ultimos_partidos(df, equipo, fecha, n=N_ULTIMOS_PARTIDOS):
    partidos = df[
        ((df["home_team"] == equipo) | (df["away_team"] == equipo)) &
        (df["date"] < fecha)
    ].sort_values("date", ascending=False).head(n)

    registros = []

    for _, fila in partidos.iterrows():
        if fila["home_team"] == equipo:
            rival = fila["away_team"]
            goles_favor = fila["home_score"]
            goles_contra = fila["away_score"]
        else:
            rival = fila["home_team"]
            goles_favor = fila["away_score"]
            goles_contra = fila["home_score"]

        if goles_favor > goles_contra:
            puntos = 3
        elif goles_favor == goles_contra:
            puntos = 1
        else:
            puntos = 0

        registros.append({
            "rival": rival,
            "goles_favor": goles_favor,
            "goles_contra": goles_contra,
            "puntos": puntos
        })

    return pd.DataFrame(registros)


def calcular_forma_ponderada(df, equipo, fecha, ranking_fifa, promedio_fifa):
    ultimos = obtener_ultimos_partidos(df, equipo, fecha)

    if ultimos.empty:
        return {
            "gf12": 1.0,
            "ga12": 1.0,
            "pts12": 1.0,
            "partidos_previos": 0
        }

    pesos_tiempo = np.array([ALPHA_TIEMPO ** k for k in range(len(ultimos))])

    pesos_rival = []

    for rival in ultimos["rival"]:
        puntos_rival = ranking_fifa.get(rival, promedio_fifa)
        pesos_rival.append(puntos_rival / promedio_fifa)

    pesos_rival = np.array(pesos_rival)
    pesos = pesos_tiempo * pesos_rival

    gf12 = np.sum(ultimos["goles_favor"] * pesos) / np.sum(pesos)
    ga12 = np.sum(ultimos["goles_contra"] * pesos) / np.sum(pesos)
    pts12 = np.sum(ultimos["puntos"] * pesos) / np.sum(pesos)

    return {
        "gf12": gf12,
        "ga12": ga12,
        "pts12": pts12,
        "partidos_previos": len(ultimos)
    }

# HISTORIAL DIRECTO

def calcular_h2h(df, equipo_a, equipo_b, fecha, n=10):
    partidos = df[
        (
            ((df["home_team"] == equipo_a) & (df["away_team"] == equipo_b)) |
            ((df["home_team"] == equipo_b) & (df["away_team"] == equipo_a))
        ) &
        (df["date"] < fecha)
    ].sort_values("date", ascending=False).head(n)

    if partidos.empty:
        return 0.0

    valores = []

    for _, fila in partidos.iterrows():
        if fila["home_team"] == equipo_a:
            gf = fila["home_score"]
            gc = fila["away_score"]
        else:
            gf = fila["away_score"]
            gc = fila["home_score"]

        if gf > gc:
            valores.append(1)
        elif gf == gc:
            valores.append(0)
        else:
            valores.append(-1)

    pesos = np.array([ALPHA_TIEMPO ** k for k in range(len(valores))])

    return float(np.sum(np.array(valores) * pesos) / np.sum(pesos))


# FEATURES DE UN PARTIDO
def construir_features_partido(
    historial,
    fecha,
    local,
    visitante,
    neutral,
    ranking_fifa,
    ranking_elo,
    promedio_fifa,
    promedio_elo
):
    forma_local = calcular_forma_ponderada(
        historial,
        local,
        fecha,
        ranking_fifa,
        promedio_fifa
    )

    forma_visitante = calcular_forma_ponderada(
        historial,
        visitante,
        fecha,
        ranking_fifa,
        promedio_fifa
    )

    fifa_local = ranking_fifa.get(local, promedio_fifa)
    fifa_visitante = ranking_fifa.get(visitante, promedio_fifa)

    elo_local = ranking_elo.get(local, promedio_elo)
    elo_visitante = ranking_elo.get(visitante, promedio_elo)

    neutral = int(neutral)
    home_advantage = 0 if neutral == 1 else 1

    h2h = calcular_h2h(historial, local, visitante, fecha)

    return {
        "home_gf12": forma_local["gf12"],
        "home_ga12": forma_local["ga12"],
        "home_pts12": forma_local["pts12"],
        "home_prev_matches": forma_local["partidos_previos"],

        "away_gf12": forma_visitante["gf12"],
        "away_ga12": forma_visitante["ga12"],
        "away_pts12": forma_visitante["pts12"],
        "away_prev_matches": forma_visitante["partidos_previos"],

        "fifa_home": fifa_local,
        "fifa_away": fifa_visitante,
        "diff_fifa": fifa_local - fifa_visitante,

        "elo_home": elo_local,
        "elo_away": elo_visitante,
        "diff_elo": elo_local - elo_visitante,

        "h2h": h2h,
        "neutral": neutral,
        "home_advantage": home_advantage
    }

# DATA DE ENTRENAMIENTO
def construir_data_modelo(resultados, fifa, elo):
    ranking_fifa = dict(zip(fifa["team"], fifa["fifa_points"]))
    ranking_elo = dict(zip(elo["team"], elo["elo_points"]))

    promedio_fifa = fifa["fifa_points"].mean()
    promedio_elo = elo["elo_points"].mean()

    historial = filtrar_partidos_oficiales(resultados)

    fecha_corte = obtener_fecha_corte_entrenamiento(historial)

    train_matches = historial[
        (historial["date"] >= pd.to_datetime(FECHA_INICIO_ENTRENAMIENTO)) &
        (historial["date"] < fecha_corte)
    ].copy()

    filas = []
    total = len(train_matches)

    print()
    print("========================================")
    print("CONSTRUYENDO DATA DEL MODELO")
    print("========================================")
    print("Fecha inicio entrenamiento:", FECHA_INICIO_ENTRENAMIENTO)
    print("Fecha corte entrenamiento:", fecha_corte.date())
    print("Fecha máxima en results.csv:", historial["date"].max().date())
    print("Partidos para entrenamiento:", total)
    print("========================================")

    for contador, (_, partido) in enumerate(train_matches.iterrows(), start=1):
        fecha = partido["date"]
        local = partido["home_team"]
        visitante = partido["away_team"]
        neutral = partido["neutral"]

        features = construir_features_partido(
            historial=historial,
            fecha=fecha,
            local=local,
            visitante=visitante,
            neutral=neutral,
            ranking_fifa=ranking_fifa,
            ranking_elo=ranking_elo,
            promedio_fifa=promedio_fifa,
            promedio_elo=promedio_elo
        )

        fila = {
            "date": fecha,
            "home_team": local,
            "away_team": visitante,
            "home_score": partido["home_score"],
            "away_score": partido["away_score"],
            "tournament": partido["tournament"]
        }

        fila.update(features)
        filas.append(fila)

        if contador % 500 == 0:
            print(f"Procesados {contador} de {total} partidos...")

    data_modelo = pd.DataFrame(filas)
    data_modelo.to_csv(SALIDA_DATA_MODELO, index=False, encoding="utf-8")

    print()
    print("Data del modelo creada:")
    print(SALIDA_DATA_MODELO)
    print("Filas:", len(data_modelo))
    print("Columnas:", len(data_modelo.columns))

    if not data_modelo.empty:
        print("Fecha mínima data_modelo:", data_modelo["date"].min())
        print("Fecha máxima data_modelo:", data_modelo["date"].max())

    return data_modelo, historial, ranking_fifa, ranking_elo, promedio_fifa, promedio_elo, fecha_corte

# FIXTURES PARA PREDICCION

def cargar_fixtures():
    if ARCHIVO_FIXTURES_GRUPOS.exists():
        fixtures = pd.read_csv(ARCHIVO_FIXTURES_GRUPOS)
    elif ARCHIVO_FIXTURES_COMPLETO.exists():
        fixtures = pd.read_csv(ARCHIVO_FIXTURES_COMPLETO)

        if "known_teams" in fixtures.columns:
            fixtures = fixtures[fixtures["known_teams"] == True].copy()
        elif "stage" in fixtures.columns:
            fixtures = fixtures[
                fixtures["stage"].astype(str).str.lower().str.contains("group", na=False)
            ].copy()
    else:
        print()
        print("AVISO:")
        print("No existe fixtures_2026_group_stage.csv ni fixtures_2026.csv.")
        print("Se omitira la creacion de data/fixtures_modelo.csv.")
        print("Para octavos usa: python 09_actualizar_fixtures_modelo_eliminatorias.py")
        return None

    fixtures.columns = fixtures.columns.str.strip().str.replace("\ufeff", "", regex=False)

    if "date" not in fixtures.columns:
        if "date_et" in fixtures.columns:
            fixtures["date"] = fixtures["date_et"]
        elif "fecha" in fixtures.columns:
            fixtures["date"] = fixtures["fecha"]
        else:
            print("Columnas encontradas en fixtures:")
            print(fixtures.columns.tolist())
            raise KeyError(
                "No se encontró columna date ni date_et en el archivo de fixtures."
            )

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.dropna(subset=["date", "home_team", "away_team"])

    fixtures["home_team"] = fixtures["home_team"].apply(normalizar_nombre_equipo)
    fixtures["away_team"] = fixtures["away_team"].apply(normalizar_nombre_equipo)

    if "neutral" in fixtures.columns:
        fixtures["neutral"] = fixtures["neutral"].apply(convertir_booleano)
    else:
        fixtures["neutral"] = 1

    return fixtures


def construir_fixtures_modelo(
    fixtures,
    historial,
    ranking_fifa,
    ranking_elo,
    promedio_fifa,
    promedio_elo,
    fecha_corte
):
    filas = []

    for _, partido in fixtures.iterrows():
        fecha_partido = partido["date"]

        # Si el partido es futuro, usa todo el historial disponible hasta la fecha de corte.
        # Si el partido ya ocurrió, usa solo historial previo a ese partido para evitar fuga.
        fecha_referencia = min(fecha_partido, fecha_corte)

        local = partido["home_team"]
        visitante = partido["away_team"]
        neutral = partido["neutral"]

        features = construir_features_partido(
            historial=historial,
            fecha=fecha_referencia,
            local=local,
            visitante=visitante,
            neutral=neutral,
            ranking_fifa=ranking_fifa,
            ranking_elo=ranking_elo,
            promedio_fifa=promedio_fifa,
            promedio_elo=promedio_elo
        )

        fila = {
            "date": fecha_partido,
            "home_team": local,
            "away_team": visitante,
            "neutral": neutral
        }

        for col in ["match_no", "stage", "group", "venue_city", "competition"]:
            if col in partido.index:
                fila[col] = partido[col]

        fila.update(features)
        filas.append(fila)

    fixtures_modelo = pd.DataFrame(filas)
    fixtures_modelo.to_csv(SALIDA_FIXTURES_MODELO, index=False, encoding="utf-8")

    print()
    print("Fixtures para prediccion creados:")
    print(SALIDA_FIXTURES_MODELO)
    print("Filas:", len(fixtures_modelo))

    return fixtures_modelo

# EJECUCION PRINCIPAL
def main():
    DATA_DIR.mkdir(exist_ok=True)

    resultados, fifa, elo = cargar_datos()

    (
        data_modelo,
        historial,
        ranking_fifa,
        ranking_elo,
        promedio_fifa,
        promedio_elo,
        fecha_corte
    ) = construir_data_modelo(resultados, fifa, elo)

    fixtures = cargar_fixtures()

    if fixtures is not None:
        construir_fixtures_modelo(
            fixtures=fixtures,
            historial=historial,
            ranking_fifa=ranking_fifa,
            ranking_elo=ranking_elo,
            promedio_fifa=promedio_fifa,
            promedio_elo=promedio_elo,
            fecha_corte=fecha_corte
        )

    print()
    print("Proceso terminado correctamente.")


if __name__ == "__main__":
    main()
