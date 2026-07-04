# ============================================================
# 06_predecir_clasificados.py
# Predice tablas de grupo, clasificados y cruces de 16avos
# Mundial 2026
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_FIXTURES = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_FAIR_PLAY = DATA_DIR / "fair_play.csv"
ARCHIVO_FIFA_RANKING = DATA_DIR / "fifa_ranking.csv"

ARCHIVO_PREDICCIONES = OUT_DIR / "predicciones_fixture.csv"
ARCHIVO_TOP10 = OUT_DIR / "top10_marcadores.csv"

ARCHIVO_PARTIDOS_USADOS = OUT_DIR / "partidos_grupo_resultado_modelo.csv"
ARCHIVO_TABLAS_GRUPOS = OUT_DIR / "tabla_grupos_predicha.csv"
ARCHIVO_CLASIFICADOS_GRUPOS = OUT_DIR / "clasificados_grupos.csv"
ARCHIVO_MEJORES_TERCEROS = OUT_DIR / "mejores_terceros.csv"
ARCHIVO_CLASIFICADOS_16AVOS = OUT_DIR / "clasificados_16avos.csv"
ARCHIVO_CRUCES_16AVOS = OUT_DIR / "cruces_16avos_predichos.csv"

IMAGEN_TABLAS_GRUPOS = OUT_DIR / "panel_tablas_grupos.png"
IMAGEN_CLASIFICADOS = OUT_DIR / "panel_clasificados_16avos.png"
IMAGEN_CRUCES_16AVOS = OUT_DIR / "panel_cruces_16avos.png"

# True: usa resultados reales si ya existen en results.csv.
# False: usa solo marcadores predichos por el modelo.
USAR_RESULTADOS_REALES = True


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

def cargar_archivos():
    if not ARCHIVO_FIXTURES.exists():
        raise FileNotFoundError("No existe data/fixtures_modelo.csv")

    if not ARCHIVO_PREDICCIONES.exists():
        raise FileNotFoundError("No existe outputs/predicciones_fixture.csv. Ejecuta primero el 02.")

    if not ARCHIVO_TOP10.exists():
        raise FileNotFoundError("No existe outputs/top10_marcadores.csv. Ejecuta primero el 02.")

    OUT_DIR.mkdir(exist_ok=True)

    fixtures = pd.read_csv(ARCHIVO_FIXTURES)
    predicciones = pd.read_csv(ARCHIVO_PREDICCIONES)
    top10 = pd.read_csv(ARCHIVO_TOP10)

    fixtures.columns = fixtures.columns.str.strip()
    predicciones.columns = predicciones.columns.str.strip()
    top10.columns = top10.columns.str.strip()

    if "match_id" not in fixtures.columns and "match_no" in fixtures.columns:
        fixtures = fixtures.rename(columns={"match_no": "match_id"})

    columnas_fixture = [
        "date",
        "match_id",
        "group",
        "home_team",
        "away_team"
    ]

    for col in columnas_fixture:
        if col not in fixtures.columns:
            print("Columnas encontradas en fixtures_modelo.csv:")
            print(fixtures.columns.tolist())
            raise ValueError("Falta columna en fixtures_modelo.csv: " + col)

    fixtures["date"] = pd.to_datetime(
        fixtures["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    predicciones["date"] = pd.to_datetime(
        predicciones["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    fixtures["match_id"] = pd.to_numeric(
        fixtures["match_id"],
        errors="coerce"
    ).astype(int)

    predicciones["match_id"] = pd.to_numeric(
        predicciones["match_id"],
        errors="coerce"
    ).astype(int)

    top10["match_id"] = pd.to_numeric(
        top10["match_id"],
        errors="coerce"
    ).astype(int)

    if ARCHIVO_RESULTS.exists():
        results = pd.read_csv(ARCHIVO_RESULTS)
        results.columns = results.columns.str.strip()

        results["date"] = pd.to_datetime(
            results["date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        results["home_score"] = pd.to_numeric(
            results["home_score"],
            errors="coerce"
        )

        results["away_score"] = pd.to_numeric(
            results["away_score"],
            errors="coerce"
        )

        results = results.dropna(subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score"
        ])
    else:
        results = pd.DataFrame()

    return fixtures, predicciones, top10, results


# ============================================================
# RESULTADOS REALES Y PREDICHOS
# ============================================================

def buscar_resultado_real(results, date, home_team, away_team):
    if results.empty:
        return None

    condicion = (
        (results["date"] == date) &
        (results["home_team"] == home_team) &
        (results["away_team"] == away_team)
    )

    if condicion.any():
        fila = results[condicion].iloc[-1]

        return {
            "home_score": int(fila["home_score"]),
            "away_score": int(fila["away_score"]),
            "fuente": "REAL"
        }

    return None


def buscar_marcador_predicho(top10, match_id):
    top_partido = top10[top10["match_id"] == match_id].copy()

    if top_partido.empty:
        return None

    if "rank" in top_partido.columns:
        top_partido = top_partido.sort_values("rank")
    else:
        top_partido = top_partido.sort_values(
            "probability_percent",
            ascending=False
        )

    fila = top_partido.iloc[0]

    score = str(fila["score"])
    partes = score.split("-")

    if len(partes) != 2:
        return None

    return {
        "home_score": int(partes[0]),
        "away_score": int(partes[1]),
        "fuente": "PREDICHO_TOP1"
    }


def buscar_marcador_por_lambdas(predicciones, match_id):
    fila = predicciones[predicciones["match_id"] == match_id]

    if fila.empty:
        return None

    fila = fila.iloc[0]

    home_score = int(round(float(fila["lambda_home"])))
    away_score = int(round(float(fila["lambda_away"])))

    return {
        "home_score": max(home_score, 0),
        "away_score": max(away_score, 0),
        "fuente": "PREDICHO_LAMBDA"
    }


def construir_partidos_con_marcador(fixtures, predicciones, top10, results):
    filas = []

    fixtures_ordenado = fixtures.sort_values(
        ["group", "date", "match_id"]
    ).reset_index(drop=True)

    for _, partido in fixtures_ordenado.iterrows():
        match_id = int(partido["match_id"])
        date = partido["date"]
        group = partido["group"]
        home_team = partido["home_team"]
        away_team = partido["away_team"]

        marcador = None

        if USAR_RESULTADOS_REALES:
            marcador = buscar_resultado_real(
                results=results,
                date=date,
                home_team=home_team,
                away_team=away_team
            )

        if marcador is None:
            marcador = buscar_marcador_predicho(
                top10=top10,
                match_id=match_id
            )

        if marcador is None:
            marcador = buscar_marcador_por_lambdas(
                predicciones=predicciones,
                match_id=match_id
            )

        if marcador is None:
            raise ValueError("No se pudo obtener marcador para match_id " + str(match_id))

        filas.append({
            "match_id": match_id,
            "date": date,
            "group": group,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": marcador["home_score"],
            "away_score": marcador["away_score"],
            "source": marcador["fuente"]
        })

    return pd.DataFrame(filas)


# ============================================================
# TABLAS DE GRUPO
# ============================================================

def inicializar_tabla(equipos, grupo):
    tabla = {}

    for equipo in equipos:
        tabla[equipo] = {
            "group": grupo,
            "team": equipo,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "points": 0,
            "source_real_matches": 0,
            "source_predicted_matches": 0
        }

    return tabla


def actualizar_tabla(tabla, home_team, away_team, home_score, away_score, fuente):
    tabla[home_team]["played"] += 1
    tabla[away_team]["played"] += 1

    tabla[home_team]["gf"] += home_score
    tabla[home_team]["ga"] += away_score

    tabla[away_team]["gf"] += away_score
    tabla[away_team]["ga"] += home_score

    tabla[home_team]["gd"] = tabla[home_team]["gf"] - tabla[home_team]["ga"]
    tabla[away_team]["gd"] = tabla[away_team]["gf"] - tabla[away_team]["ga"]

    if fuente == "REAL":
        tabla[home_team]["source_real_matches"] += 1
        tabla[away_team]["source_real_matches"] += 1
    else:
        tabla[home_team]["source_predicted_matches"] += 1
        tabla[away_team]["source_predicted_matches"] += 1

    if home_score > away_score:
        tabla[home_team]["points"] += 3
        tabla[home_team]["wins"] += 1
        tabla[away_team]["losses"] += 1

    elif home_score < away_score:
        tabla[away_team]["points"] += 3
        tabla[away_team]["wins"] += 1
        tabla[home_team]["losses"] += 1

    else:
        tabla[home_team]["points"] += 1
        tabla[away_team]["points"] += 1
        tabla[home_team]["draws"] += 1
        tabla[away_team]["draws"] += 1

    return tabla


def construir_tablas_grupos(partidos):
    tablas = []

    grupos = sorted(partidos["group"].dropna().unique())

    for grupo in grupos:
        partidos_grupo = partidos[partidos["group"] == grupo].copy()

        equipos = sorted(
            set(partidos_grupo["home_team"].tolist())
            | set(partidos_grupo["away_team"].tolist())
        )

        tabla = inicializar_tabla(equipos=equipos, grupo=grupo)

        for _, partido in partidos_grupo.iterrows():
            tabla = actualizar_tabla(
                tabla=tabla,
                home_team=partido["home_team"],
                away_team=partido["away_team"],
                home_score=int(partido["home_score"]),
                away_score=int(partido["away_score"]),
                fuente=partido["source"]
            )

        tabla_grupo = pd.DataFrame(tabla.values())
        tablas.append(tabla_grupo)

    tabla_total = pd.concat(tablas, ignore_index=True)

    return tabla_total


# ============================================================
# FAIR PLAY Y RANKING FIFA
# ============================================================

def cargar_fair_play():
    if not ARCHIVO_FAIR_PLAY.exists():
        return pd.DataFrame(columns=[
            "team",
            "yellow_cards",
            "indirect_red_cards",
            "direct_red_cards",
            "fair_play_score"
        ])

    fair = pd.read_csv(ARCHIVO_FAIR_PLAY)
    fair.columns = fair.columns.str.strip()

    if "team" not in fair.columns:
        raise ValueError("fair_play.csv debe tener una columna llamada team")

    if "yellow_cards" not in fair.columns:
        fair["yellow_cards"] = 0

    if "indirect_red_cards" not in fair.columns:
        fair["indirect_red_cards"] = 0

    if "direct_red_cards" not in fair.columns:
        fair["direct_red_cards"] = 0

    fair["yellow_cards"] = pd.to_numeric(
        fair["yellow_cards"],
        errors="coerce"
    ).fillna(0)

    fair["indirect_red_cards"] = pd.to_numeric(
        fair["indirect_red_cards"],
        errors="coerce"
    ).fillna(0)

    fair["direct_red_cards"] = pd.to_numeric(
        fair["direct_red_cards"],
        errors="coerce"
    ).fillna(0)

    fair["fair_play_score"] = (
        fair["yellow_cards"] * -1
        + fair["indirect_red_cards"] * -3
        + fair["direct_red_cards"] * -4
    )

    return fair[[
        "team",
        "yellow_cards",
        "indirect_red_cards",
        "direct_red_cards",
        "fair_play_score"
    ]]


def cargar_fifa_strength_desde_fixtures(fixtures):
    filas = []

    if "fifa_home" in fixtures.columns:
        home = fixtures[["home_team", "fifa_home"]].copy()
        home.columns = ["team", "fifa_strength"]
        filas.append(home)

    if "fifa_away" in fixtures.columns:
        away = fixtures[["away_team", "fifa_away"]].copy()
        away.columns = ["team", "fifa_strength"]
        filas.append(away)

    if len(filas) == 0:
        return pd.DataFrame(columns=["team", "fifa_strength"])

    fifa = pd.concat(filas, ignore_index=True)

    fifa["fifa_strength"] = pd.to_numeric(
        fifa["fifa_strength"],
        errors="coerce"
    )

    fifa = fifa.dropna(subset=["fifa_strength"])

    fifa = fifa.groupby("team", as_index=False)["fifa_strength"].mean()

    return fifa


def cargar_fifa_strength(fixtures):
    if ARCHIVO_FIFA_RANKING.exists():
        ranking = pd.read_csv(ARCHIVO_FIFA_RANKING)
        ranking.columns = ranking.columns.str.strip()

        posibles_team = [
            "team",
            "country",
            "country_full",
            "name",
            "team_name"
        ]

        posibles_points = [
            "points",
            "total_points",
            "fifa_points",
            "rank_points"
        ]

        posibles_rank = [
            "rank",
            "ranking",
            "fifa_rank"
        ]

        col_team = None
        col_points = None
        col_rank = None

        for col in posibles_team:
            if col in ranking.columns:
                col_team = col
                break

        for col in posibles_points:
            if col in ranking.columns:
                col_points = col
                break

        for col in posibles_rank:
            if col in ranking.columns:
                col_rank = col
                break

        if col_team is not None and col_points is not None:
            fifa = ranking[[col_team, col_points]].copy()
            fifa.columns = ["team", "fifa_strength"]

            fifa["fifa_strength"] = pd.to_numeric(
                fifa["fifa_strength"],
                errors="coerce"
            )

            fifa = fifa.dropna(subset=["fifa_strength"])
            return fifa

        if col_team is not None and col_rank is not None:
            fifa = ranking[[col_team, col_rank]].copy()
            fifa.columns = ["team", "fifa_rank"]

            fifa["fifa_rank"] = pd.to_numeric(
                fifa["fifa_rank"],
                errors="coerce"
            )

            fifa = fifa.dropna(subset=["fifa_rank"])

            # En ranking FIFA, menor puesto es mejor.
            # Lo convertimos a fuerza: rank 1 = -1, rank 20 = -20.
            fifa["fifa_strength"] = fifa["fifa_rank"] * -1

            return fifa[["team", "fifa_strength"]]

    return cargar_fifa_strength_desde_fixtures(fixtures)


def agregar_desempates_fifa(tablas_grupos, fixtures):
    fair = cargar_fair_play()
    fifa = cargar_fifa_strength(fixtures)

    tablas = tablas_grupos.copy()

    tablas = tablas.merge(
        fair,
        on="team",
        how="left"
    )

    tablas = tablas.merge(
        fifa,
        on="team",
        how="left"
    )

    columnas_fair = [
        "yellow_cards",
        "indirect_red_cards",
        "direct_red_cards",
        "fair_play_score"
    ]

    for col in columnas_fair:
        if col not in tablas.columns:
            tablas[col] = 0

        tablas[col] = pd.to_numeric(
            tablas[col],
            errors="coerce"
        ).fillna(0)

    if "fifa_strength" not in tablas.columns:
        tablas["fifa_strength"] = 0

    tablas["fifa_strength"] = pd.to_numeric(
        tablas["fifa_strength"],
        errors="coerce"
    ).fillna(0)

    return tablas


def recalcular_posiciones_grupo_con_desempates(tablas_grupos):
    grupos_recalculados = []

    for grupo in sorted(tablas_grupos["group"].unique()):
        tabla = tablas_grupos[tablas_grupos["group"] == grupo].copy()

        tabla = tabla.sort_values(
            by=[
                "points",
                "gd",
                "gf",
                "fair_play_score",
                "fifa_strength",
                "team"
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True
            ]
        ).reset_index(drop=True)

        tabla["position"] = tabla.index + 1

        columnas = [
            "group",
            "position",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "gf",
            "ga",
            "gd",
            "points",
            "yellow_cards",
            "indirect_red_cards",
            "direct_red_cards",
            "fair_play_score",
            "fifa_strength",
            "source_real_matches",
            "source_predicted_matches"
        ]

        tabla = tabla[columnas]

        grupos_recalculados.append(tabla)

    return pd.concat(grupos_recalculados, ignore_index=True)


# ============================================================
# CLASIFICADOS
# ============================================================

def obtener_clasificados(tablas_grupos):
    tablas_grupos = tablas_grupos.copy()

    if "position" not in tablas_grupos.columns and "position_group" in tablas_grupos.columns:
        tablas_grupos = tablas_grupos.rename(columns={"position_group": "position"})

    columnas_requeridas = [
        "group",
        "team",
        "position",
        "points",
        "gd",
        "gf",
    ]

    columnas_faltantes = [
        col for col in columnas_requeridas
        if col not in tablas_grupos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "No se puede calcular clasificados. Faltan columnas en tablas_grupos: "
            + ", ".join(columnas_faltantes)
        )

    if "fair_play_score" not in tablas_grupos.columns:
        tablas_grupos["fair_play_score"] = 0

    if "fifa_strength" not in tablas_grupos.columns:
        tablas_grupos["fifa_strength"] = 0

    primeros = tablas_grupos[tablas_grupos["position"] == 1].copy()
    segundos = tablas_grupos[tablas_grupos["position"] == 2].copy()
    terceros = tablas_grupos[tablas_grupos["position"] == 3].copy()

    primeros["classification_type"] = "PRIMERO_GRUPO"
    segundos["classification_type"] = "SEGUNDO_GRUPO"
    terceros["classification_type"] = "TERCERO_GRUPO"

    # Regla para mejores terceros:
    # 1. Puntos
    # 2. Diferencia de goles
    # 3. Goles a favor
    # 4. Fair Play
    # 5. Ranking FIFA
    mejores_terceros = terceros.sort_values(
        by=[
            "points",
            "gd",
            "gf",
            "fair_play_score",
            "fifa_strength",
            "team"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    mejores_terceros["third_place_rank"] = mejores_terceros.index + 1
    mejores_terceros["advance"] = mejores_terceros["third_place_rank"] <= 8

    mejores_terceros_clasificados = mejores_terceros[
        mejores_terceros["advance"] == True
    ].copy()

    mejores_terceros_clasificados["classification_type"] = "MEJOR_TERCERO"

    clasificados_16avos = pd.concat([
        primeros,
        segundos,
        mejores_terceros_clasificados
    ], ignore_index=True)

    clasificados_grupos = pd.concat([
        primeros,
        segundos
    ], ignore_index=True)

    clasificados_grupos = clasificados_grupos.sort_values(
        by=["group", "position"]
    ).reset_index(drop=True)

    clasificados_16avos = clasificados_16avos.sort_values(
        by=["classification_type", "group"]
    ).reset_index(drop=True)

    return clasificados_grupos, mejores_terceros, clasificados_16avos


# ============================================================
# CRUCES SIMULADOS DE 16AVOS
# ============================================================

def obtener_equipo_slot(clasificados_16avos, slot):
    tipo = slot[0]
    grupo = slot[1]

    if tipo == "1":
        filtro = (
            (clasificados_16avos["group"] == grupo) &
            (clasificados_16avos["position"] == 1)
        )

    elif tipo == "2":
        filtro = (
            (clasificados_16avos["group"] == grupo) &
            (clasificados_16avos["position"] == 2)
        )

    else:
        return "TBD"

    if filtro.any():
        return clasificados_16avos[filtro].iloc[0]["team"]

    return "TBD"


def asignar_mejor_tercero(mejores_terceros, grupos_permitidos, usados):
    candidatos = mejores_terceros[
        (mejores_terceros["advance"] == True) &
        (mejores_terceros["group"].isin(grupos_permitidos))
    ].copy()

    candidatos = candidatos.sort_values(
        by=["third_place_rank"]
    )

    for _, fila in candidatos.iterrows():
        equipo = fila["team"]

        if equipo not in usados:
            usados.add(equipo)
            return equipo, fila["group"]

    return "TBD", ""


def construir_cruces_16avos(clasificados_16avos, mejores_terceros):
    usados_terceros = set()

    template = [
        ("16avos 1", "2A", "2B", []),
        ("16avos 2", "1C", "2F", []),
        ("16avos 3", "1E", "3X", ["A", "B", "C", "D", "F"]),
        ("16avos 4", "1F", "2C", []),
        ("16avos 5", "2E", "2I", []),
        ("16avos 6", "1I", "3X", ["C", "D", "F", "G", "H"]),
        ("16avos 7", "1A", "3X", ["C", "E", "F", "H", "I"]),
        ("16avos 8", "1L", "3X", ["E", "H", "I", "J", "K"]),
        ("16avos 9", "1G", "3X", ["A", "E", "H", "I", "J"]),
        ("16avos 10", "1D", "3X", ["B", "E", "F", "I", "J"]),
        ("16avos 11", "1H", "2J", []),
        ("16avos 12", "2K", "2L", []),
        ("16avos 13", "1B", "3X", ["E", "F", "G", "I", "J"]),
        ("16avos 14", "2D", "2G", []),
        ("16avos 15", "1J", "2H", []),
        ("16avos 16", "1K", "3X", ["D", "E", "I", "J", "L"]),
    ]

    filas = []

    for nombre_partido, slot_1, slot_2, grupos_tercero in template:
        equipo_1 = obtener_equipo_slot(clasificados_16avos, slot_1)

        if slot_2 == "3X":
            equipo_2, grupo_tercero = asignar_mejor_tercero(
                mejores_terceros=mejores_terceros,
                grupos_permitidos=grupos_tercero,
                usados=usados_terceros
            )

            slot_2_final = "3" + grupo_tercero if grupo_tercero != "" else "3X"

        else:
            equipo_2 = obtener_equipo_slot(clasificados_16avos, slot_2)
            slot_2_final = slot_2

        filas.append({
            "partido_16avos": nombre_partido,
            "slot_1": slot_1,
            "equipo_1": equipo_1,
            "slot_2": slot_2_final,
            "equipo_2": equipo_2
        })

    return pd.DataFrame(filas)


# ============================================================
# IMPRESION EN CONSOLA
# ============================================================

def mostrar_tablas_por_grupo(tablas_grupos):
    print()
    print("========================================")
    print("TABLAS FINALES PREDICHAS POR GRUPO")
    print("========================================")

    for grupo in sorted(tablas_grupos["group"].unique()):
        tabla = tablas_grupos[tablas_grupos["group"] == grupo].copy()

        print()
        print("GRUPO", grupo)
        print(tabla[[
            "position",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "gf",
            "ga",
            "gd",
            "points"
        ]].to_string(index=False))


def mostrar_clasificados(clasificados_grupos, mejores_terceros, clasificados_16avos):
    print()
    print("========================================")
    print("PRIMEROS Y SEGUNDOS DE GRUPO")
    print("========================================")
    print(clasificados_grupos[[
        "group",
        "position",
        "team",
        "points",
        "gd",
        "gf",
        "fair_play_score",
        "fifa_strength",
        "classification_type"
    ]].to_string(index=False))

    print()
    print("========================================")
    print("RANKING DE TERCEROS")
    print("========================================")
    print(mejores_terceros[[
        "third_place_rank",
        "group",
        "team",
        "points",
        "gd",
        "gf",
        "fair_play_score",
        "fifa_strength",
        "advance"
    ]].to_string(index=False))

    print()
    print("========================================")
    print("CLASIFICADOS A 16AVOS")
    print("========================================")
    print(clasificados_16avos[[
        "group",
        "position",
        "team",
        "points",
        "gd",
        "gf",
        "classification_type"
    ]].to_string(index=False))


# ============================================================
# GRAFICOS / VENTANAS
# ============================================================

def graficar_tablas_grupos(tablas_grupos):
    grupos = sorted(tablas_grupos["group"].unique())

    fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=False)
    axes = axes.flatten()

    fig.suptitle(
        "Mundial 2026 - Tablas finales predichas por grupo",
        fontsize=20,
        fontweight="bold"
    )

    for i, grupo in enumerate(grupos):
        ax = axes[i]
        ax.axis("off")

        tabla = tablas_grupos[tablas_grupos["group"] == grupo].copy()

        tabla_visual = tabla[[
            "position",
            "team",
            "points",
            "gd",
            "gf"
        ]].copy()

        tabla_visual.columns = [
            "Pos",
            "Equipo",
            "Pts",
            "DG",
            "GF"
        ]

        tabla_plot = ax.table(
            cellText=tabla_visual.values,
            colLabels=tabla_visual.columns,
            cellLoc="center",
            loc="center"
        )

        tabla_plot.auto_set_font_size(False)
        tabla_plot.set_fontsize(9)
        tabla_plot.scale(1, 1.5)

        ax.set_title(
            "Grupo " + str(grupo),
            fontsize=13,
            fontweight="bold"
        )

    for j in range(len(grupos), len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=0.03,
        top=0.92,
        wspace=0.20,
        hspace=0.35
    )

    fig.savefig(
        IMAGEN_TABLAS_GRUPOS,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen de tablas de grupos guardada:", IMAGEN_TABLAS_GRUPOS)

    plt.show()


def graficar_clasificados_16avos(clasificados_16avos, mejores_terceros):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=False)

    fig.suptitle(
        "Mundial 2026 - Clasificados predichos a 16avos",
        fontsize=20,
        fontweight="bold"
    )

    primeros = clasificados_16avos[
        clasificados_16avos["classification_type"] == "PRIMERO_GRUPO"
    ].copy()

    primeros = primeros[[
        "group",
        "team",
        "points",
        "gd"
    ]]

    primeros.columns = [
        "Grupo",
        "Equipo",
        "Pts",
        "DG"
    ]

    axes[0].axis("off")
    tabla1 = axes[0].table(
        cellText=primeros.values,
        colLabels=primeros.columns,
        cellLoc="center",
        loc="center"
    )

    tabla1.auto_set_font_size(False)
    tabla1.set_fontsize(9)
    tabla1.scale(1, 1.5)
    axes[0].set_title("Primeros de grupo", fontsize=13, fontweight="bold")

    segundos = clasificados_16avos[
        clasificados_16avos["classification_type"] == "SEGUNDO_GRUPO"
    ].copy()

    segundos = segundos[[
        "group",
        "team",
        "points",
        "gd"
    ]]

    segundos.columns = [
        "Grupo",
        "Equipo",
        "Pts",
        "DG"
    ]

    axes[1].axis("off")
    tabla2 = axes[1].table(
        cellText=segundos.values,
        colLabels=segundos.columns,
        cellLoc="center",
        loc="center"
    )

    tabla2.auto_set_font_size(False)
    tabla2.set_fontsize(9)
    tabla2.scale(1, 1.5)
    axes[1].set_title("Segundos de grupo", fontsize=13, fontweight="bold")

    terceros = mejores_terceros[
        mejores_terceros["advance"] == True
    ].copy()

    terceros = terceros[[
        "third_place_rank",
        "group",
        "team",
        "points",
        "gd"
    ]]

    terceros.columns = [
        "Rank",
        "Grupo",
        "Equipo",
        "Pts",
        "DG"
    ]

    axes[2].axis("off")
    tabla3 = axes[2].table(
        cellText=terceros.values,
        colLabels=terceros.columns,
        cellLoc="center",
        loc="center"
    )

    tabla3.auto_set_font_size(False)
    tabla3.set_fontsize(9)
    tabla3.scale(1, 1.5)
    axes[2].set_title("Mejores terceros", fontsize=13, fontweight="bold")

    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=0.05,
        top=0.86,
        wspace=0.25
    )

    fig.savefig(
        IMAGEN_CLASIFICADOS,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen de clasificados guardada:", IMAGEN_CLASIFICADOS)

    plt.show()


def graficar_cruces_16avos(cruces_16avos):
    fig, ax = plt.subplots(figsize=(16, 13), constrained_layout=False)
    ax.axis("off")

    fig.suptitle(
        "Mundial 2026 - Simulación de cruces de 16avos",
        fontsize=20,
        fontweight="bold"
    )

    tabla_visual = cruces_16avos[[
        "partido_16avos",
        "slot_1",
        "equipo_1",
        "slot_2",
        "equipo_2"
    ]].copy()

    tabla_visual.columns = [
        "Partido",
        "Slot 1",
        "Equipo 1",
        "Slot 2",
        "Equipo 2"
    ]

    tabla_plot = ax.table(
        cellText=tabla_visual.values,
        colLabels=tabla_visual.columns,
        cellLoc="center",
        loc="center"
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(10)
    tabla_plot.scale(1, 1.6)

    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=0.03,
        top=0.90
    )

    fig.savefig(
        IMAGEN_CRUCES_16AVOS,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen de cruces de 16avos guardada:", IMAGEN_CRUCES_16AVOS)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    fixtures, predicciones, top10, results = cargar_archivos()

    partidos_finales = construir_partidos_con_marcador(
        fixtures=fixtures,
        predicciones=predicciones,
        top10=top10,
        results=results
    )

    tablas_grupos = construir_tablas_grupos(partidos_finales)

    tablas_grupos = agregar_desempates_fifa(
        tablas_grupos=tablas_grupos,
        fixtures=fixtures
    )

    tablas_grupos = recalcular_posiciones_grupo_con_desempates(
        tablas_grupos=tablas_grupos
    )

    resultado_clasificados = obtener_clasificados(
        tablas_grupos=tablas_grupos
    )

    if not isinstance(resultado_clasificados, tuple) or len(resultado_clasificados) != 3:
        raise ValueError(
            "obtener_clasificados debe devolver exactamente 3 elementos: "
            "clasificados_grupos, mejores_terceros y clasificados_16avos."
        )

    clasificados_grupos = resultado_clasificados[0]
    mejores_terceros = resultado_clasificados[1]
    clasificados_16avos = resultado_clasificados[2]

    cruces_16avos = construir_cruces_16avos(
        clasificados_16avos=clasificados_16avos,
        mejores_terceros=mejores_terceros
    )

    partidos_finales.to_csv(
        ARCHIVO_PARTIDOS_USADOS,
        index=False,
        encoding="utf-8"
    )

    tablas_grupos.to_csv(
        ARCHIVO_TABLAS_GRUPOS,
        index=False,
        encoding="utf-8"
    )

    clasificados_grupos.to_csv(
        ARCHIVO_CLASIFICADOS_GRUPOS,
        index=False,
        encoding="utf-8"
    )

    mejores_terceros.to_csv(
        ARCHIVO_MEJORES_TERCEROS,
        index=False,
        encoding="utf-8"
    )

    clasificados_16avos.to_csv(
        ARCHIVO_CLASIFICADOS_16AVOS,
        index=False,
        encoding="utf-8"
    )

    cruces_16avos.to_csv(
        ARCHIVO_CRUCES_16AVOS,
        index=False,
        encoding="utf-8"
    )

    mostrar_tablas_por_grupo(tablas_grupos)

    mostrar_clasificados(
        clasificados_grupos=clasificados_grupos,
        mejores_terceros=mejores_terceros,
        clasificados_16avos=clasificados_16avos
    )

    print()
    print("========================================")
    print("CRUCES SIMULADOS DE 16AVOS")
    print("========================================")
    print(cruces_16avos.to_string(index=False))

    graficar_tablas_grupos(tablas_grupos)

    graficar_clasificados_16avos(
        clasificados_16avos=clasificados_16avos,
        mejores_terceros=mejores_terceros
    )

    graficar_cruces_16avos(cruces_16avos)

    print()
    print("========================================")
    print("ARCHIVOS GENERADOS")
    print("========================================")
    print(ARCHIVO_PARTIDOS_USADOS)
    print(ARCHIVO_TABLAS_GRUPOS)
    print(ARCHIVO_CLASIFICADOS_GRUPOS)
    print(ARCHIVO_MEJORES_TERCEROS)
    print(ARCHIVO_CLASIFICADOS_16AVOS)
    print(ARCHIVO_CRUCES_16AVOS)
    print(IMAGEN_TABLAS_GRUPOS)
    print(IMAGEN_CLASIFICADOS)
    print(IMAGEN_CRUCES_16AVOS)
    print("========================================")


if __name__ == "__main__":
    main()
