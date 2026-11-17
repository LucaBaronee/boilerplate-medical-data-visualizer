import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1) Importamos los datos del CSV
df = pd.read_csv('medical_examination.csv')

# 2) IMC = peso / (altura en metros)^2. Si es mayor a 25 -> sobrepeso (1), si no -> 0
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (bmi > 25).astype(int)

# 3) Normalizamos: 0 = bueno, 1 = malo. Si el valor es 1 queda en 0; si es mayor a 1 pasa a 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4) Gráfico categórico
def draw_cat_plot():
    # 5) Pasamos las columnas a formato "largo": una fila por cada (paciente, variable, valor)
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6) Agrupamos por cardio, variable y valor, y contamos cuántos hay en cada grupo
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7) Dibujamos un gráfico de barras: un panel por cada valor de cardio
    graph = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8) Obtenemos la figura del gráfico
    fig = graph.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10) Mapa de calor
def draw_heat_map():
    # 11) Limpiamos los datos incorrectos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12) Calculamos la matriz de correlación
    corr = df_heat.corr()

    # 13) Máscara para tapar el triángulo superior (es un espejo del inferior)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14) Creamos la figura y los ejes de matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15) Dibujamos el mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
