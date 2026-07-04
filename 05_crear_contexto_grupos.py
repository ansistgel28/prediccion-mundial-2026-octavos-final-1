# ============================================================
# 05_crear_contexto_grupos.py
# Contexto competitivo mejorado para fase de grupos Mundial 2026
# ============================================================

import itertools
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ARCHIVO_FIXTURES = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_SALIDA = DATA_DIR / "fixtures_contexto_grupos.csv"


# ============================================================
# FUNCIONES BASICAS DE TABLA
# ============================================================

def inicializar_tabla(equipos):
    tabla = {}

    for equipo in equipos:
        tabla[equipo] = {
            "team": equipo,
            "points": 0,
            "played": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }

    return tabla


def copiar_tabla(tabla):
    nueva = {}

    for equipo, datos in tabla.items():
        nueva[equipo] = datos.copy()

    return nueva


def actualizar_tabla(tabla, home_team, away_team, home_score, away_score):
    tabla[home_team]["played"] += 1
    tabla[away_team]["played"] += 1

    tabla[home_team]["gf"] += home_score
    tabla[home_team]["ga"] += away_score

    tabla[away_team]["gf"] += away_score
    tabla[away_team]["ga"] += home_score

    tabla[home_team]["gd"] = tabla[home_team]["gf"] - tabla[home_team]["ga"]
    tabla[away_team]["gd"] = tabla[away_team]["gf"] - tabla[away_team]["ga"]

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


def ordenar_tabla(tabla):
    df = pd.DataFrame(tabla.values())

    df = df.sort_values(
        by=["points", "gd", "gf", "team"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    df["position"] = df.index + 1

    return df


def obtener_fila_equipo(tabla_df, equipo):
    return tabla_df[tabla_df["team"] == equipo].iloc[0]


# ============================================================
# DETECCION DE JORNADA
# ============================================================

def detectar_matchday(partidos_grupo):
    partidos_grupo = partidos_grupo.sort_values(["date", "match_id"]).copy()

    contador = {}
    matchdays = []

    for _, fila in partidos_grupo.iterrows():
        home = fila["home_team"]
        away = fila["away_team"]

        contador.setdefault(home, 0)
        contador.setdefault(away, 0)

        jornada = max(contador[home], contador[away]) + 1
        matchdays.append(jornada)

        contador[home] += 1
        contador[away] += 1

    partidos_grupo["matchday_group"] = matchdays

    return partidos_grupo


# ============================================================
# BUSQUEDA DE RESULTADOS REALES
# ============================================================

def buscar_resultado(results, date, home_team, away_team):
    condicion = (
        (results["date"] == date) &
        (results["home_team"] == home_team) &
        (results["away_team"] == away_team)
    )

    if condicion.any():
        r = results[condicion].iloc[-1]
        return int(r["home_score"]), int(r["away_score"])

    return None


# ============================================================
# SIMULACION DE ESCENARIOS DE GRUPO
# ============================================================

def aplicar_resultado_nominal(tabla, home_team, away_team, resultado):
    # H = gana local, D = empate, A = gana visitante
    if resultado == "H":
        actualizar_tabla(tabla, home_team, away_team, 1, 0)
    elif resultado == "D":
        actualizar_tabla(tabla, home_team, away_team, 1, 1)
    elif resultado == "A":
        actualizar_tabla(tabla, home_team, away_team, 0, 1)

    return tabla


def resultado_equipo_en_partido(partido, resultado, equipo):
    home = partido["home_team"]
    away = partido["away_team"]

    if equipo == home:
        if resultado == "H":
            return "win"
        elif resultado == "D":
            return "draw"
        else:
            return "loss"

    if equipo == away:
        if resultado == "A":
            return "win"
        elif resultado == "D":
            return "draw"
        else:
            return "loss"

    return None


def simular_jornada(tabla_base, jornada_partidos):
    partidos = jornada_partidos.to_dict("records")
    resultados_posibles = ["H", "D", "A"]

    simulaciones = []

    for combinacion in itertools.product(resultados_posibles, repeat=len(partidos)):
        tabla_simulada = copiar_tabla(tabla_base)
        resultados_partidos = {}

        for partido, resultado in zip(partidos, combinacion):
            match_id = int(partido["match_id"])

            resultados_partidos[match_id] = resultado

            tabla_simulada = aplicar_resultado_nominal(
                tabla=tabla_simulada,
                home_team=partido["home_team"],
                away_team=partido["away_team"],
                resultado=resultado
            )

        tabla_final = ordenar_tabla(tabla_simulada)

        simulaciones.append({
            "tabla": tabla_final,
            "resultados": resultados_partidos
        })

    return simulaciones


def calcular_tasa_top2(posiciones):
    if len(posiciones) == 0:
        return 0.0

    cantidad = sum(1 for p in posiciones if p <= 2)

    return cantidad / len(posiciones)


def calcular_tasa_primero(posiciones):
    if len(posiciones) == 0:
        return 0.0

    cantidad = sum(1 for p in posiciones if p == 1)

    return cantidad / len(posiciones)


# ============================================================
# ANALISIS COMPETITIVO DEL EQUIPO
# ============================================================

def analizar_contexto_equipo(tabla_base, jornada_partidos, equipo, matchday):
    tabla_actual = ordenar_tabla(tabla_base)
    fila_actual = obtener_fila_equipo(tabla_actual, equipo)

    points = int(fila_actual["points"])
    played = int(fila_actual["played"])
    gf = int(fila_actual["gf"])
    ga = int(fila_actual["ga"])
    gd = int(fila_actual["gd"])
    position = int(fila_actual["position"])

    partido_equipo = jornada_partidos[
        (jornada_partidos["home_team"] == equipo) |
        (jornada_partidos["away_team"] == equipo)
    ]

    if partido_equipo.empty:
        return {
            "points_group": points,
            "goal_diff_group": gd,
            "goals_for_group": gf,
            "goals_against_group": ga,
            "matches_played_group": played,
            "position_group": position,
            "possible_best_position": position,
            "possible_worst_position": position,
            "qualify_rate": 0.0,
            "first_place_rate": 0.0,
            "qualify_rate_if_win": 0.0,
            "qualify_rate_if_draw": 0.0,
            "qualify_rate_if_loss": 0.0,
            "first_rate_if_win": 0.0,
            "first_rate_if_draw": 0.0,
            "first_rate_if_loss": 0.0,
            "already_qualified": 0,
            "eliminated": 0,
            "must_win": 0,
            "need_goal_difference": 0,
            "rotation_risk": 0,
            "playing_for_first": 0,
            "risk_drop_to_third": 0,
            "needs_result": 0,
            "context_note": "Sin partido en esta jornada"
        }

    partido_equipo = partido_equipo.iloc[0].to_dict()
    match_id_equipo = int(partido_equipo["match_id"])

    simulaciones = simular_jornada(tabla_base, jornada_partidos)

    posiciones_todas = []
    posiciones_win = []
    posiciones_draw = []
    posiciones_loss = []

    for simulacion in simulaciones:
        tabla_simulada = simulacion["tabla"]
        fila_simulada = obtener_fila_equipo(tabla_simulada, equipo)

        posicion_final = int(fila_simulada["position"])
        posiciones_todas.append(posicion_final)

        resultado_match = simulacion["resultados"][match_id_equipo]

        resultado_equipo = resultado_equipo_en_partido(
            partido=partido_equipo,
            resultado=resultado_match,
            equipo=equipo
        )

        if resultado_equipo == "win":
            posiciones_win.append(posicion_final)
        elif resultado_equipo == "draw":
            posiciones_draw.append(posicion_final)
        elif resultado_equipo == "loss":
            posiciones_loss.append(posicion_final)

    possible_best_position = min(posiciones_todas)
    possible_worst_position = max(posiciones_todas)

    qualify_rate = calcular_tasa_top2(posiciones_todas)
    first_place_rate = calcular_tasa_primero(posiciones_todas)

    qualify_rate_if_win = calcular_tasa_top2(posiciones_win)
    qualify_rate_if_draw = calcular_tasa_top2(posiciones_draw)
    qualify_rate_if_loss = calcular_tasa_top2(posiciones_loss)

    first_rate_if_win = calcular_tasa_primero(posiciones_win)
    first_rate_if_draw = calcular_tasa_primero(posiciones_draw)
    first_rate_if_loss = calcular_tasa_primero(posiciones_loss)

    guaranteed_top2 = qualify_rate == 1.0
    can_finish_top2 = qualify_rate > 0.0

    guaranteed_first = first_place_rate == 1.0
    can_finish_first = first_place_rate > 0.0

    risk_drop_to_third = 1 if possible_worst_position >= 3 else 0

    playing_for_first = 0
    if matchday >= 2 and can_finish_first and not guaranteed_first:
        playing_for_first = 1

    already_qualified = 1 if guaranteed_top2 else 0
    eliminated = 1 if not can_finish_top2 else 0

    needs_result = 0
    if can_finish_top2 and not guaranteed_top2:
        needs_result = 1

    if playing_for_first == 1:
        needs_result = 1

    must_win = 0

    # Segunda fecha: si llega sin ganar, tiene presión
    if matchday == 2:
        if points == 0:
            must_win = 1
            needs_result = 1

        elif points <= 1 and position >= 3:
            needs_result = 1

    # Tercera fecha: análisis de escenarios reales de grupo
    if matchday >= 3:
        if can_finish_top2 and not guaranteed_top2:
            if qualify_rate_if_win > qualify_rate_if_draw:
                must_win = 1

            if qualify_rate_if_draw < 1.0 and position >= 3:
                must_win = 1

            if qualify_rate_if_win > 0 and qualify_rate_if_draw == 0:
                must_win = 1

    need_goal_difference = 0

    if matchday >= 3:
        if must_win == 1 and risk_drop_to_third == 1:
            need_goal_difference = 1

        elif needs_result == 1 and gd <= 0 and position >= 3:
            need_goal_difference = 1

    rotation_risk = 0

    # Solo se permite rotación si está realmente seguro
    # y no pelea primer lugar ni corre riesgo de caer
    if matchday >= 3:
        if already_qualified == 1 and playing_for_first == 0 and risk_drop_to_third == 0:
            rotation_risk = 1

    elif matchday == 2:
        if points >= 6 and playing_for_first == 0 and risk_drop_to_third == 0:
            rotation_risk = 1

    # Nota explicativa
    if rotation_risk == 1:
        context_note = "Puede rotar: clasificacion muy segura"
    elif must_win == 1 and need_goal_difference == 1:
        context_note = "Necesita ganar y cuidar diferencia de goles"
    elif must_win == 1:
        context_note = "Necesita ganar para mejorar clasificacion"
    elif needs_result == 1 and playing_for_first == 1:
        context_note = "Necesita resultado y pelea el primer lugar"
    elif needs_result == 1:
        context_note = "Necesita resultado para no complicarse"
    elif playing_for_first == 1:
        context_note = "Pelea el primer lugar del grupo"
    elif already_qualified == 1:
        context_note = "Clasificacion asegurada, pero sin rotacion fuerte"
    elif eliminated == 1:
        context_note = "Sin opcion clara de top 2"
    else:
        context_note = "Presion competitiva normal"

    return {
        "points_group": points,
        "goal_diff_group": gd,
        "goals_for_group": gf,
        "goals_against_group": ga,
        "matches_played_group": played,
        "position_group": position,

        "possible_best_position": possible_best_position,
        "possible_worst_position": possible_worst_position,

        "qualify_rate": round(qualify_rate, 4),
        "first_place_rate": round(first_place_rate, 4),

        "qualify_rate_if_win": round(qualify_rate_if_win, 4),
        "qualify_rate_if_draw": round(qualify_rate_if_draw, 4),
        "qualify_rate_if_loss": round(qualify_rate_if_loss, 4),

        "first_rate_if_win": round(first_rate_if_win, 4),
        "first_rate_if_draw": round(first_rate_if_draw, 4),
        "first_rate_if_loss": round(first_rate_if_loss, 4),

        "already_qualified": int(already_qualified),
        "eliminated": int(eliminated),
        "must_win": int(must_win),
        "need_goal_difference": int(need_goal_difference),
        "rotation_risk": int(rotation_risk),

        "playing_for_first": int(playing_for_first),
        "risk_drop_to_third": int(risk_drop_to_third),
        "needs_result": int(needs_result),

        "context_note": context_note
    }


# ============================================================
# NORMALIZACION DE ARCHIVOS
# ============================================================

def preparar_fixtures(fixtures):
    fixtures.columns = fixtures.columns.str.strip()

    if "match_id" not in fixtures.columns and "match_no" in fixtures.columns:
        fixtures = fixtures.rename(columns={"match_no": "match_id"})

    columnas_necesarias = [
        "date",
        "home_team",
        "away_team",
        "group",
        "match_id"
    ]

    for col in columnas_necesarias:
        if col not in fixtures.columns:
            print("Columnas encontradas en fixtures:")
            print(fixtures.columns.tolist())
            raise ValueError("Falta columna en fixtures_modelo.csv: " + col)

    fixtures["date"] = pd.to_datetime(
        fixtures["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    fixtures["match_id"] = pd.to_numeric(
        fixtures["match_id"],
        errors="coerce"
    ).astype(int)

    fixtures = fixtures.dropna(subset=[
        "date",
        "home_team",
        "away_team",
        "group",
        "match_id"
    ])

    return fixtures


def preparar_results(results):
    results.columns = results.columns.str.strip()

    columnas_necesarias = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ]

    for col in columnas_necesarias:
        if col not in results.columns:
            print("Columnas encontradas en results:")
            print(results.columns.tolist())
            raise ValueError("Falta columna en results.csv: " + col)

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

    return results


# ============================================================
# CONSTRUCCION DEL CONTEXTO
# ============================================================

def construir_contexto():
    if not ARCHIVO_FIXTURES.exists():
        raise FileNotFoundError("No existe: " + str(ARCHIVO_FIXTURES))

    if not ARCHIVO_RESULTS.exists():
        raise FileNotFoundError("No existe: " + str(ARCHIVO_RESULTS))

    fixtures = pd.read_csv(ARCHIVO_FIXTURES)
    results = pd.read_csv(ARCHIVO_RESULTS)

    fixtures = preparar_fixtures(fixtures)
    results = preparar_results(results)

    fixtures = fixtures.sort_values(
        ["group", "date", "match_id"]
    ).reset_index(drop=True)

    filas_contexto = []

    grupos = sorted(fixtures["group"].dropna().unique())

    for grupo in grupos:
        partidos_grupo = fixtures[fixtures["group"] == grupo].copy()
        partidos_grupo = detectar_matchday(partidos_grupo)

        equipos = sorted(
            set(partidos_grupo["home_team"].tolist())
            | set(partidos_grupo["away_team"].tolist())
        )

        tabla = inicializar_tabla(equipos)

        jornadas = sorted(partidos_grupo["matchday_group"].unique())

        for matchday in jornadas:
            jornada_partidos = partidos_grupo[
                partidos_grupo["matchday_group"] == matchday
            ].copy()

            # IMPORTANTE:
            # El contexto de todos los partidos de la misma jornada se calcula
            # antes de actualizar resultados de esa jornada.
            # Así evitamos usar el resultado de un partido simultáneo.
            for _, partido in jornada_partidos.iterrows():
                home_team = partido["home_team"]
                away_team = partido["away_team"]

                home_estado = analizar_contexto_equipo(
                    tabla_base=tabla,
                    jornada_partidos=jornada_partidos,
                    equipo=home_team,
                    matchday=int(matchday)
                )

                away_estado = analizar_contexto_equipo(
                    tabla_base=tabla,
                    jornada_partidos=jornada_partidos,
                    equipo=away_team,
                    matchday=int(matchday)
                )

                fila = partido.to_dict()
                fila["matchday_group"] = int(matchday)

                for clave, valor in home_estado.items():
                    fila["home_" + clave] = valor

                for clave, valor in away_estado.items():
                    fila["away_" + clave] = valor

                filas_contexto.append(fila)

            # Ahora sí se actualiza la tabla con resultados reales de esa jornada
            for _, partido in jornada_partidos.iterrows():
                resultado = buscar_resultado(
                    results=results,
                    date=partido["date"],
                    home_team=partido["home_team"],
                    away_team=partido["away_team"]
                )

                if resultado is not None:
                    home_score, away_score = resultado

                    tabla = actualizar_tabla(
                        tabla=tabla,
                        home_team=partido["home_team"],
                        away_team=partido["away_team"],
                        home_score=home_score,
                        away_score=away_score
                    )

    df_contexto = pd.DataFrame(filas_contexto)

    df_contexto = df_contexto.sort_values(
        ["group", "matchday_group", "match_id"]
    ).reset_index(drop=True)

    df_contexto.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("CONTEXTO COMPETITIVO MEJORADO CREADO")
    print("========================================")
    print("Archivo:", ARCHIVO_SALIDA)
    print("Filas:", len(df_contexto))
    print("Columnas:", len(df_contexto.columns))
    print("========================================")

    columnas_mostrar = [
        "match_id",
        "date",
        "group",
        "home_team",
        "away_team",
        "matchday_group",

        "home_points_group",
        "home_position_group",
        "home_possible_best_position",
        "home_possible_worst_position",
        "home_already_qualified",
        "home_must_win",
        "home_needs_result",
        "home_playing_for_first",
        "home_risk_drop_to_third",
        "home_need_goal_difference",
        "home_rotation_risk",
        "home_context_note",

        "away_points_group",
        "away_position_group",
        "away_possible_best_position",
        "away_possible_worst_position",
        "away_already_qualified",
        "away_must_win",
        "away_needs_result",
        "away_playing_for_first",
        "away_risk_drop_to_third",
        "away_need_goal_difference",
        "away_rotation_risk",
        "away_context_note",
    ]

    columnas_mostrar = [
        col for col in columnas_mostrar if col in df_contexto.columns
    ]

    print()
    print(df_contexto[columnas_mostrar].to_string(index=False))


def main():
    construir_contexto()


if __name__ == "__main__":
    main()