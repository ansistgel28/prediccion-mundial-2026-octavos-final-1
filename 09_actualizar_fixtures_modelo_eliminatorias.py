# ============================================================
# 09_actualizar_fixtures_modelo_eliminatorias.py
# Actualiza results.csv con 16avos y construye fixtures modelo
# para simular octavos usando resultados recientes.
# ============================================================

import shutil
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_FIXTURES_MODELO = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_FIXTURES_16AVOS = DATA_DIR / "fixtures_16avos.csv"
ARCHIVO_FIXTURES_OCTAVOS = DATA_DIR / "fixtures_octavos.csv"

ARCHIVO_RESULTS_BACKUP = DATA_DIR / "results_backup_antes_eliminatorias.csv"
ARCHIVO_RESULTS_ACTUALIZADO = DATA_DIR / "results_actualizado_eliminatorias.csv"

ARCHIVO_FIXTURES_16AVOS_MODELO = DATA_DIR / "fixtures_16avos_modelo.csv"
ARCHIVO_FIXTURES_OCTAVOS_MODELO = DATA_DIR / "fixtures_octavos_modelo.csv"
ARCHIVO_FIXTURES_MODELO_ACTUALIZADO = DATA_DIR / "fixtures_modelo_actualizado.csv"

N_ULTIMOS_PARTIDOS = 12


# ============================================================
# UTILIDADES
# ============================================================

def leer_csv(ruta):
    if not ruta.exists():
        raise FileNotFoundError("No existe el archivo: " + str(ruta))

    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()

    return df


def normalizar_fecha(df, columna="date"):
    df[columna] = pd.to_datetime(
        df[columna],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    return df


def convertir_numero(df, columna):
    if columna in df.columns:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    return df


# ============================================================
# ACTUALIZAR results.csv CON PARTIDOS DE 16AVOS
# ============================================================

def construir_filas_resultados_desde_16avos(fixtures_16avos, columnas_results):
    filas = []

    for _, fila in fixtures_16avos.iterrows():
        home_score = pd.to_numeric(fila.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(fila.get("away_score"), errors="coerce")

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        nuevo = {}

        for col in columnas_results:
            nuevo[col] = ""

        nuevo["date"] = fila["date"]
        nuevo["home_team"] = fila["home_team"]
        nuevo["away_team"] = fila["away_team"]
        nuevo["home_score"] = int(home_score)
        nuevo["away_score"] = int(away_score)

        if "tournament" in columnas_results:
            nuevo["tournament"] = "FIFA World Cup"

        if "neutral" in columnas_results:
            nuevo["neutral"] = True

        if "city" in columnas_results:
            nuevo["city"] = "Unknown"

        if "country" in columnas_results:
            nuevo["country"] = "United States/Canada/Mexico"

        filas.append(nuevo)

    return pd.DataFrame(filas)


def actualizar_results_con_16avos(results, fixtures_16avos):
    columnas_results = results.columns.tolist()

    nuevas_filas = construir_filas_resultados_desde_16avos(
        fixtures_16avos=fixtures_16avos,
        columnas_results=columnas_results
    )

    if nuevas_filas.empty:
        return results.copy()

    results_actualizado = results.copy()

    for _, fila in nuevas_filas.iterrows():
        condicion = (
            (results_actualizado["date"] == fila["date"]) &
            (results_actualizado["home_team"] == fila["home_team"]) &
            (results_actualizado["away_team"] == fila["away_team"])
        )

        if condicion.any():
            idx = results_actualizado[condicion].index[-1]
            results_actualizado.loc[idx, "home_score"] = fila["home_score"]
            results_actualizado.loc[idx, "away_score"] = fila["away_score"]

            if "tournament" in results_actualizado.columns:
                results_actualizado.loc[idx, "tournament"] = "FIFA World Cup"

            if "neutral" in results_actualizado.columns:
                results_actualizado.loc[idx, "neutral"] = True

        else:
            results_actualizado = pd.concat(
                [results_actualizado, pd.DataFrame([fila])],
                ignore_index=True
            )

    results_actualizado = results_actualizado.sort_values(
        by=["date", "home_team", "away_team"]
    ).reset_index(drop=True)

    return results_actualizado


# ============================================================
# MAPAS FIFA / ELO DESDE fixtures_modelo.csv
# ============================================================

def construir_mapas_ranking(fixtures_modelo):
    filas_fifa = []
    filas_elo = []

    if "fifa_home" in fixtures_modelo.columns:
        home = fixtures_modelo[["home_team", "fifa_home"]].copy()
        home.columns = ["team", "valor"]
        filas_fifa.append(home)

    if "fifa_away" in fixtures_modelo.columns:
        away = fixtures_modelo[["away_team", "fifa_away"]].copy()
        away.columns = ["team", "valor"]
        filas_fifa.append(away)

    if "elo_home" in fixtures_modelo.columns:
        home = fixtures_modelo[["home_team", "elo_home"]].copy()
        home.columns = ["team", "valor"]
        filas_elo.append(home)

    if "elo_away" in fixtures_modelo.columns:
        away = fixtures_modelo[["away_team", "elo_away"]].copy()
        away.columns = ["team", "valor"]
        filas_elo.append(away)

    if len(filas_fifa) > 0:
        fifa = pd.concat(filas_fifa, ignore_index=True)
        fifa["valor"] = pd.to_numeric(fifa["valor"], errors="coerce")
        fifa = fifa.dropna(subset=["valor"])
        fifa = fifa.groupby("team", as_index=False)["valor"].mean()
        mapa_fifa = dict(zip(fifa["team"], fifa["valor"]))
    else:
        mapa_fifa = {}

    if len(filas_elo) > 0:
        elo = pd.concat(filas_elo, ignore_index=True)
        elo["valor"] = pd.to_numeric(elo["valor"], errors="coerce")
        elo = elo.dropna(subset=["valor"])
        elo = elo.groupby("team", as_index=False)["valor"].mean()
        mapa_elo = dict(zip(elo["team"], elo["valor"]))
    else:
        mapa_elo = {}

    return mapa_fifa, mapa_elo


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def preparar_results(results):
    results = results.copy()

    results = normalizar_fecha(results, "date")

    for col in ["home_score", "away_score"]:
        results = convertir_numero(results, col)

    results = results.dropna(subset=[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ])

    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)

    results = results.sort_values("date").reset_index(drop=True)

    return results


def obtener_partidos_equipo(results, equipo, fecha_corte):
    fecha_corte = pd.to_datetime(fecha_corte)

    df = results.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["date_dt"] < fecha_corte].copy()

    condicion = (
        (df["home_team"] == equipo) |
        (df["away_team"] == equipo)
    )

    df = df[condicion].copy()
    df = df.sort_values("date_dt")

    filas = []

    for _, partido in df.iterrows():
        if partido["home_team"] == equipo:
            gf = int(partido["home_score"])
            ga = int(partido["away_score"])
        else:
            gf = int(partido["away_score"])
            ga = int(partido["home_score"])

        if gf > ga:
            pts = 3
        elif gf == ga:
            pts = 1
        else:
            pts = 0

        filas.append({
            "date": partido["date"],
            "team": equipo,
            "gf": gf,
            "ga": ga,
            "pts": pts
        })

    return pd.DataFrame(filas)


def calcular_forma_reciente(results, equipo, fecha_partido):
    partidos = obtener_partidos_equipo(
        results=results,
        equipo=equipo,
        fecha_corte=fecha_partido
    )

    if partidos.empty:
        return {
            "gf12": 0,
            "ga12": 0,
            "pts12": 0,
            "prev_matches": 0
        }

    ultimos = partidos.tail(N_ULTIMOS_PARTIDOS).copy()

    return {
        "gf12": int(ultimos["gf"].sum()),
        "ga12": int(ultimos["ga"].sum()),
        "pts12": int(ultimos["pts"].sum()),
        "prev_matches": int(len(ultimos))
    }


def calcular_h2h(results, home_team, away_team, fecha_partido):
    fecha_corte = pd.to_datetime(fecha_partido)

    df = results.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["date_dt"] < fecha_corte].copy()

    condicion = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )

    h2h = df[condicion].copy().sort_values("date_dt").tail(5)

    if h2h.empty:
        return 0

    puntos_home_equipo = 0
    puntos_away_equipo = 0

    for _, partido in h2h.iterrows():
        if partido["home_team"] == home_team:
            goles_home_equipo = int(partido["home_score"])
            goles_away_equipo = int(partido["away_score"])
        else:
            goles_home_equipo = int(partido["away_score"])
            goles_away_equipo = int(partido["home_score"])

        if goles_home_equipo > goles_away_equipo:
            puntos_home_equipo += 3
        elif goles_home_equipo < goles_away_equipo:
            puntos_away_equipo += 3
        else:
            puntos_home_equipo += 1
            puntos_away_equipo += 1

    return puntos_home_equipo - puntos_away_equipo


def construir_fixture_modelo(fixture, results, mapa_fifa, mapa_elo, stage):
    filas = []

    fixture = fixture.copy()
    fixture = normalizar_fecha(fixture, "date")

    if "match_id" not in fixture.columns and "match_no" in fixture.columns:
        fixture = fixture.rename(columns={"match_no": "match_id"})

    for _, partido in fixture.iterrows():
        fecha = partido["date"]
        home_team = partido["home_team"]
        away_team = partido["away_team"]

        forma_home = calcular_forma_reciente(
            results=results,
            equipo=home_team,
            fecha_partido=fecha
        )

        forma_away = calcular_forma_reciente(
            results=results,
            equipo=away_team,
            fecha_partido=fecha
        )

        fifa_home = float(mapa_fifa.get(home_team, 0))
        fifa_away = float(mapa_fifa.get(away_team, 0))
        elo_home = float(mapa_elo.get(home_team, 0))
        elo_away = float(mapa_elo.get(away_team, 0))

        fila = {
            "date": fecha,
            "home_team": home_team,
            "away_team": away_team,
            "neutral": 1,
            "match_id": int(partido["match_id"]),
            "match_no": int(partido["match_id"]),
            "stage": partido.get("round", stage),
            "group": "",
            "venue_city": partido.get("venue_city", ""),
            "competition": "FIFA World Cup",
            "home_gf12": forma_home["gf12"],
            "home_ga12": forma_home["ga12"],
            "home_pts12": forma_home["pts12"],
            "home_prev_matches": forma_home["prev_matches"],
            "away_gf12": forma_away["gf12"],
            "away_ga12": forma_away["ga12"],
            "away_pts12": forma_away["pts12"],
            "away_prev_matches": forma_away["prev_matches"],
            "fifa_home": fifa_home,
            "fifa_away": fifa_away,
            "diff_fifa": fifa_home - fifa_away,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "diff_elo": elo_home - elo_away,
            "h2h": calcular_h2h(
                results=results,
                home_team=home_team,
                away_team=away_team,
                fecha_partido=fecha
            ),
            "home_advantage": 0,
            "home_score": partido.get("home_score", ""),
            "away_score": partido.get("away_score", ""),
            "pen_home": partido.get("pen_home", ""),
            "pen_away": partido.get("pen_away", ""),
            "source_note": partido.get("source_note", "")
        }

        filas.append(fila)

    return pd.DataFrame(filas)


# ============================================================
# MAIN
# ============================================================

def main():
    results = leer_csv(ARCHIVO_RESULTS)
    fixtures_modelo = leer_csv(ARCHIVO_FIXTURES_MODELO)
    fixtures_16avos = leer_csv(ARCHIVO_FIXTURES_16AVOS)
    fixtures_octavos = leer_csv(ARCHIVO_FIXTURES_OCTAVOS)

    results = preparar_results(results)
    fixtures_16avos = normalizar_fecha(fixtures_16avos, "date")
    fixtures_octavos = normalizar_fecha(fixtures_octavos, "date")

    if not ARCHIVO_RESULTS_BACKUP.exists():
        shutil.copy2(ARCHIVO_RESULTS, ARCHIVO_RESULTS_BACKUP)

    results_actualizado = actualizar_results_con_16avos(
        results=results,
        fixtures_16avos=fixtures_16avos
    )

    results_actualizado = preparar_results(results_actualizado)

    results_actualizado.to_csv(
        ARCHIVO_RESULTS_ACTUALIZADO,
        index=False,
        encoding="utf-8"
    )

    # Sobrescribe results.csv para que los demás scripts ya usen los 16avos.
    results_actualizado.to_csv(
        ARCHIVO_RESULTS,
        index=False,
        encoding="utf-8"
    )

    mapa_fifa, mapa_elo = construir_mapas_ranking(
        fixtures_modelo=fixtures_modelo
    )

    fixtures_16avos_modelo = construir_fixture_modelo(
        fixture=fixtures_16avos,
        results=results,
        mapa_fifa=mapa_fifa,
        mapa_elo=mapa_elo,
        stage="16AVOS"
    )

    fixtures_octavos_modelo = construir_fixture_modelo(
        fixture=fixtures_octavos,
        results=results_actualizado,
        mapa_fifa=mapa_fifa,
        mapa_elo=mapa_elo,
        stage="OCTAVOS"
    )

    fixtures_16avos_modelo.to_csv(
        ARCHIVO_FIXTURES_16AVOS_MODELO,
        index=False,
        encoding="utf-8"
    )

    fixtures_octavos_modelo.to_csv(
        ARCHIVO_FIXTURES_OCTAVOS_MODELO,
        index=False,
        encoding="utf-8"
    )

    # Archivo acumulado: fase de grupos + 16avos + octavos.
    fixtures_modelo_actualizado = pd.concat(
        [
            fixtures_modelo,
            fixtures_16avos_modelo,
            fixtures_octavos_modelo
        ],
        ignore_index=True,
        sort=False
    )

    fixtures_modelo_actualizado.to_csv(
        ARCHIVO_FIXTURES_MODELO_ACTUALIZADO,
        index=False,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("ACTUALIZACION COMPLETADA")
    print("========================================")
    print("Resultados actualizados:", ARCHIVO_RESULTS)
    print("Backup de results:", ARCHIVO_RESULTS_BACKUP)
    print("Copia results actualizado:", ARCHIVO_RESULTS_ACTUALIZADO)
    print("Fixtures 16avos modelo:", ARCHIVO_FIXTURES_16AVOS_MODELO)
    print("Fixtures octavos modelo:", ARCHIVO_FIXTURES_OCTAVOS_MODELO)
    print("Fixtures modelo acumulado:", ARCHIVO_FIXTURES_MODELO_ACTUALIZADO)
    print()
    print("Ultimo match_id en fixtures_modelo_actualizado:")
    print(fixtures_modelo_actualizado["match_id"].max())
    print("========================================")


if __name__ == "__main__":
    main()
