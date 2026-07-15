# Modelo predictivo para los 8vos de final del Mundial 2026

Este repositorio contiene la entrega correspondiente al **modelo predictivo para los 8vos de final del Mundial 2026**.

El objetivo principal es simular los partidos de octavos de final, estimar posibles marcadores, calcular probabilidades de clasificación a cuartos de final y analizar el riesgo disciplinario mediante datos de Fair Play y tarjetas.

---

## Integrantes del grupo

- Jean Frank Bustamante Vela
- Miguel Angel Marreros Cortegana
- Valentin Fernandez Campos
- Frank Salon Trigoso

---

## Entrega

**Modelo predictivo para los 8vos de final del Mundial 2026.**

Esta entrega se enfoca únicamente en la fase de **octavos de final**, es decir, en la predicción de los partidos que definen qué selecciones clasifican a **cuartos de final**.

---

## Enlaces individuales de GitHub

- Jean Frank Bustamante Vela:
- Miguel Angel Marreros Cortegana:
- Valentin Fernandez Campos: colocar enlace
- Frank Salon Trigoso: colocar enlace
- Maria Carmen Tuesta Chuquizuta: colocar enlace

---

## Descripción general

El proyecto utiliza modelos de regresión y aprendizaje supervisado para estimar los goles esperados de cada selección en los partidos de octavos de final.

A partir de esas estimaciones se construye una matriz de probabilidades de marcadores mediante una distribución de Poisson ajustada con corrección Dixon-Coles. También se incorpora información de Fair Play para estimar posibles tarjetas por equipo, considerando disciplina acumulada, rival, contexto de eliminatoria y dificultad del partido.

---

## Objetivo general

Construir un modelo predictivo para los partidos de octavos de final del Mundial 2026, capaz de estimar marcadores probables, probabilidades de clasificación a cuartos y posibles tarjetas por equipo.

---

## Objetivos específicos

- Actualizar la base de datos con resultados reales de 16avos de final.
- Generar variables actualizadas para los partidos de octavos.
- Estimar goles esperados mediante modelos de regresión.
- Calcular probabilidades de marcadores mediante Poisson.
- Ajustar marcadores bajos mediante Dixon-Coles.
- Estimar clasificados a cuartos de final.
- Analizar posibles tarjetas usando datos de Fair Play.
- Mostrar métricas de los modelos utilizados.

---

## Metodología resumida

```text
1. Resultados históricos y recientes.
2. Actualización con resultados de 16avos.
3. Construcción de fixtures para octavos.
4. Generación de variables predictoras.
5. Uso del modelo entrenado.
6. Cálculo de lambdas ofensivas.
7. Aplicación de Poisson + Dixon-Coles.
8. Predicción de marcadores.
9. Predicción de clasificados a cuartos.
10. Estimación de tarjetas.
11. Visualización de resultados.
```

---

## Modelos evaluados

En el proyecto se evaluaron cinco modelos:

```text
1. Regresión Lineal
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Poisson Regressor
```

Las métricas se encuentran en:

```text
outputs/competencia_modelos.csv
outputs/resumen_metricas_modelos.csv
outputs/mejor_modelo.txt
```

Para esta entrega no se recomienda volver a entrenar, porque el entrenamiento puede demorar. Se usan los modelos ya entrenados y guardados.

---

## Modelo probabilístico

Para convertir los goles esperados en probabilidades de marcadores, se usa una matriz de Poisson:

```text
P(Home = a) = (e^(-λ_home) * λ_home^a) / a!
P(Away = b) = (e^(-λ_away) * λ_away^b) / b!
P(a,b) = P(Home=a) * P(Away=b)
```

Donde:

```text
λ_home = goles esperados del equipo local
λ_away = goles esperados del equipo visitante
```

Luego se aplica la corrección Dixon-Coles para ajustar los marcadores bajos:

```text
0-0
1-0
0-1
1-1
```

Esto evita resultados poco realistas y mejora la interpretación de marcadores frecuentes en fútbol.

---

## Predicción de tarjetas

El modelo estima posibles tarjetas por equipo usando:

```text
- Tarjetas amarillas.
- Tarjetas rojas indirectas.
- Tarjetas rojas directas.
- Puntaje Fair Play.
- Cantidad de partidos jugados.
- Contexto de eliminatoria.
- Partido parejo.
- Probabilidad alta de empate.
- Diferencia FIFA/Elo entre equipos.
```

Archivo principal:

```text
data/fair_play.csv
```

Salida generada:

```text
outputs/prediccion_tarjetas_octavos.csv
```

---

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── 08_predecir_cuartos.py
├── 09_actualizar_fixtures_modelo_eliminatorias.py
├── 10_ver_metricas_modelos.py
├── data
│   ├── results.csv
│   ├── data_modelo.csv
│   ├── fair_play.csv
│   ├── fixtures_16avos.csv
│   ├── fixtures_16avos_modelo.csv
│   ├── fixtures_octavos.csv
│   ├── fixtures_octavos_modelo.csv
│   ├── fixtures_modelo_actualizado.csv
│   └── results_actualizado_eliminatorias.csv
└── outputs
    ├── competencia_modelos.csv
    ├── mejor_modelo.txt
    ├── resumen_metricas_modelos.csv
    ├── prediccion_octavos_a_cuartos.csv
    ├── top10_marcadores_octavos.csv
    ├── prediccion_tarjetas_octavos.csv
    ├── clasificados_cuartos.csv
    ├── panel_metricas_modelos.png
    └── panel_octavos_marcadores_tarjetas.png
```

---

## Scripts principales

### 08_predecir_cuartos.py

Script principal de la entrega.

Predice los partidos de octavos y determina qué selecciones clasifican a cuartos de final.

Genera:

```text
outputs/prediccion_octavos_a_cuartos.csv
outputs/top10_marcadores_octavos.csv
outputs/prediccion_tarjetas_octavos.csv
outputs/clasificados_cuartos.csv
outputs/panel_octavos_marcadores_tarjetas.png
```

---

### 09_actualizar_fixtures_modelo_eliminatorias.py

Actualiza la información del modelo con los resultados de 16avos y genera los fixtures de octavos con variables actualizadas.

Genera:

```text
data/results_actualizado_eliminatorias.csv
data/fixtures_16avos_modelo.csv
data/fixtures_octavos_modelo.csv
data/fixtures_modelo_actualizado.csv
```

---

### 10_ver_metricas_modelos.py

Muestra las métricas de los modelos ya entrenados sin volver a entrenar.

Lee:

```text
outputs/competencia_modelos.csv
outputs/mejor_modelo.txt
```

Genera:

```text
outputs/resumen_metricas_modelos.csv
outputs/panel_metricas_modelos.png
```

---

# Instrucciones para evaluación del profesor

## 1. Clonar el repositorio

```bash
git clone https://github.com/Jeanki07/prediccion-mundial-2026-octavos-final.git
cd prediccion-mundial-2026-octavos-final
```

---

## 2. Crear entorno virtual

En Linux o macOS:

```bash
python -m venv venv
source venv/bin/activate
```

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 4. Ejecutar actualización de fixtures

```bash
python 09_actualizar_fixtures_modelo_eliminatorias.py
```

Este paso actualiza los datos de eliminatorias y prepara los partidos de octavos para la predicción.

---

## 5. Ejecutar predicción de octavos a cuartos

```bash
python 08_predecir_cuartos.py
```

Este script genera:

```text
- Predicción de clasificados a cuartos.
- Top 10 marcadores probables por partido.
- Probabilidades de victoria, empate y clasificación.
- Predicción de tarjetas por selección.
- Panel visual de resultados.
```

---

## 6. Ver métricas de los modelos

```bash
python 10_ver_metricas_modelos.py
```

Este script no entrena los modelos. Solo muestra las métricas guardadas para una evaluación rápida.

---

## Orden recomendado de ejecución

```bash
python 09_actualizar_fixtures_modelo_eliminatorias.py
python 08_predecir_cuartos.py
python 10_ver_metricas_modelos.py
```

---

## Resultados generados

Después de ejecutar los scripts, revisar la carpeta:

```text
outputs/
```

Archivos principales:

```text
outputs/prediccion_octavos_a_cuartos.csv
outputs/top10_marcadores_octavos.csv
outputs/prediccion_tarjetas_octavos.csv
outputs/clasificados_cuartos.csv
outputs/resumen_metricas_modelos.csv
outputs/panel_octavos_marcadores_tarjetas.png
outputs/panel_metricas_modelos.png
```

---

## Ver resultados desde terminal

### Predicción principal

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/prediccion_octavos_a_cuartos.csv'); print(df.to_string(index=False))"
```

### Top 10 marcadores probables

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/top10_marcadores_octavos.csv'); print(df.to_string(index=False))"
```

### Predicción de tarjetas

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/prediccion_tarjetas_octavos.csv'); print(df.to_string(index=False))"
```

### Métricas de modelos

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/resumen_metricas_modelos.csv'); print(df.to_string(index=False))"
```

---

## Visualización de resultados

En Linux:

```bash
xdg-open outputs/panel_octavos_marcadores_tarjetas.png
xdg-open outputs/panel_metricas_modelos.png
```

En Windows o macOS, abrir manualmente los archivos `.png` desde la carpeta `outputs`.

---

## Interpretación de archivos de salida

### outputs/prediccion_octavos_a_cuartos.csv

Contiene:

```text
- Partido.
- Equipos.
- Lambdas del modelo.
- Probabilidad de victoria local.
- Probabilidad de empate.
- Probabilidad de victoria visitante.
- Probabilidad de avance de cada equipo.
- Marcador más probable.
- Ganador predicho.
```

---

### outputs/top10_marcadores_octavos.csv

Contiene los 10 marcadores más probables para cada partido.

Ejemplo de marcadores esperados:

```text
1-0
1-1
0-1
2-1
0-0
2-0
1-2
```

---

### outputs/prediccion_tarjetas_octavos.csv

Contiene:

```text
- Tarjetas esperadas del equipo local.
- Tarjetas esperadas del equipo visitante.
- Total esperado de tarjetas.
- Riesgo disciplinario del partido.
- Contexto de eliminatoria.
- Contexto de partido parejo.
```

---

### outputs/clasificados_cuartos.csv

Contiene las selecciones que el modelo predice como clasificadas a cuartos de final.

---

### outputs/resumen_metricas_modelos.csv

Contiene la comparación de los modelos entrenados.

---

## Nota sobre entrenamiento

El entrenamiento completo se realizó previamente con el script:

```text
02_entrenar_predecir_modelos.py
```

Ese script puede demorar porque entrena los cinco modelos. Por eso, para la evaluación rápida de esta entrega se recomienda usar:

```bash
python 10_ver_metricas_modelos.py
```

Este script muestra las métricas guardadas sin volver a entrenar.

---

## Nota importante

Este repositorio se mantiene como entrega específica para los **8vos de final del Mundial 2026**.  
No se recomienda actualizarlo con fases posteriores, para conservar la versión evaluable correspondiente a esta fase.

---

## Comando rápido de evaluación

En Linux o macOS:

```bash
git clone https://github.com/Jeanki07/prediccion-mundial-2026-octavos-final.git
cd prediccion-mundial-2026-octavos-final
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 09_actualizar_fixtures_modelo_eliminatorias.py
python 08_predecir_cuartos.py
python 10_ver_metricas_modelos.py
```

En Windows, reemplazar:

```bash
source venv/bin/activate
```

por:

```bash
venv\Scripts\activate
```
# Clasificados de 16avos a octavos de final

| N.º | Partido de 16avos | Resultado | Clasificado a octavos |
|---:|---|:---:|---|
| 1 | South Africa vs. Canada | 0-1 | Canada |
| 2 | Brazil vs. Japan | 2-1 | Brazil |
| 3 | Germany vs. Paraguay | 1-1 (3-4 en penales) | Paraguay |
| 4 | Netherlands vs. Morocco | 1-1 (2-3 en penales) | Morocco |
| 5 | Ivory Coast vs. Norway | 1-2 | Norway |
| 6 | France vs. Sweden | 3-0 | France |
| 7 | Mexico vs. Ecuador | 2-0 | Mexico |
| 8 | England vs. DR Congo | 2-1 | England |
| 9 | Belgium vs. Senegal | 3-2 | Belgium |
| 10 | United States vs. Bosnia and Herzegovina | 2-0 | United States |
| 11 | Spain vs. Austria | 3-0 | Spain |
| 12 | Portugal vs. Croatia | 2-1 | Portugal |
| 13 | Switzerland vs. Algeria | 2-0 | Switzerland |
| 14 | Australia vs. Egypt | 1-1 (2-4 en penales) | Egypt |
| 15 | Argentina vs. Cape Verde | 3-2 | Argentina |
| 16 | Colombia vs. Ghana | 1-0 | Colombia |

<img width="1354" height="323" alt="image" src="https://github.com/user-attachments/assets/5f340536-9eb0-46a4-b3fe-483cf5ed8fe7" />
