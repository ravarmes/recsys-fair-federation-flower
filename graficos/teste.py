import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dados do primeiro script
data_rgrp_activity_1 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.000981994, 0.000238225, 0.000504384, 0.000430394, 0.000188408, 3.596e-06,
        0.000134395, 0.000476567, 0.000709533, 0.000955670, 0.001092158, 0.001086757,
        0.001186913, 0.001348532, 0.001563960, 0.001856400, 0.002241598, 0.002600655,
        0.003121024, 0.003715837, 0.004516703, 0.005286370, 0.006090524, 0.006575472,
        0.007342683
    ]
}

data_rgrp_age_1 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.003396680190163166, 0.003883012178654958, 0.0032389046429972, 0.0033415192432913977,
        0.003622837286258953, 0.004139421103580343, 0.004522999975863927, 0.004788569180378254,
        0.004456227536866096, 0.004230801833006068, 0.0039457243582087425, 0.003498848430346345,
        0.003322188539265435, 0.0033420332522783544, 0.0036406929257304195, 0.003894552862944088,
        0.004579852897645133, 0.005006362570400268, 0.005874461016865226, 0.006780175495069189,
        0.00815280980914856, 0.00949522482741473, 0.010576392752614999, 0.01133786831348063,
        0.01261870780876268
    ]
}

data_rgrp_gender_1 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        9.39706e-06, 0.000111572, 9.23792e-05, 0.000143523, 0.000352234, 0.000953561,
        0.00190807, 0.00279572, 0.00327143, 0.00362377, 0.00381965, 0.00389468,
        0.00402243, 0.00412658, 0.00431023, 0.00446346, 0.00467865, 0.00481244,
        0.00503336, 0.00527175, 0.00548703, 0.00571105, 0.00594851, 0.00605597, 
        0.00620417
    ]
}

# Dados do segundo script
data_rgrp_activity_2 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.000587136, 0.000615752, 0.000647992, 0.000541797, 0.000279922, 4.5065e-05,
        2.5759e-05, 0.000224664, 0.000412887, 0.000608475, 0.000791455, 0.000906242,
        0.001140012, 0.001342053, 0.001488482, 0.001724044, 0.001940908, 0.002485204,
        0.003083606, 0.003613637, 0.003848011, 0.004577462, 0.005158144, 0.005699375,
        0.005658691
    ]
}

data_rgrp_age_2 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.003316993264715157, 0.003251807793377859, 0.0032322931654841904, 0.0035272411871037092,
        0.003970825756020964, 0.004037469233907881, 0.004009808075451978, 0.003936471667819949,
        0.004081532651030693, 0.003948377749837722, 0.004491508798822626, 0.004060774858755394,
        0.004578062433915229, 0.0043192492932716655, 0.004689115637215251, 0.004699278789104403,
        0.005361180749725294, 0.0062233703201010635, 0.007333330523745089, 0.008324128550660479,
        0.008918793010997456, 0.01062044611971569, 0.01171140963725474, 0.01326773884178339,
        0.013325354506972096
    ]
}

data_rgrp_gender_2 = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        4.06835e-05, 0.000123035, 9.77093e-05, 0.000114679, 0.000199495, 0.000520983,
        0.001176677, 0.001961658, 0.002466697, 0.002917075, 0.003144436, 0.003394917,
        0.003508075, 0.003750874, 0.003827650, 0.004021368, 0.004059270, 0.004354577,
        0.004580626, 0.004821832, 0.004893485, 0.005136513, 0.005289502, 0.005387192,
        0.005368775
    ]
}

# Criar DataFrames para todos os conjuntos de dados
df_rgrp_activity_1 = pd.DataFrame(data_rgrp_activity_1)
df_rgrp_age_1 = pd.DataFrame(data_rgrp_age_1)
df_rgrp_gender_1 = pd.DataFrame(data_rgrp_gender_1)

df_rgrp_activity_2 = pd.DataFrame(data_rgrp_activity_2)
df_rgrp_age_2 = pd.DataFrame(data_rgrp_age_2)
df_rgrp_gender_2 = pd.DataFrame(data_rgrp_gender_2)

# Criar figura com 2 linhas e 3 colunas
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Definir escalas para cada tipo de gráfico
# Atividade
y_min_activity = min(np.min(df_rgrp_activity_1['Rgrp']), np.min(df_rgrp_activity_2['Rgrp']))
y_max_activity = max(np.max(df_rgrp_activity_1['Rgrp']), np.max(df_rgrp_activity_2['Rgrp']))

# Idade
y_min_age = min(np.min(df_rgrp_age_1['Rgrp']), np.min(df_rgrp_age_2['Rgrp']))
y_max_age = max(np.max(df_rgrp_age_1['Rgrp']), np.max(df_rgrp_age_2['Rgrp']))

# Gênero
y_min_gender = min(np.min(df_rgrp_gender_1['Rgrp']), np.min(df_rgrp_gender_2['Rgrp']))
y_max_gender = max(np.max(df_rgrp_gender_1['Rgrp']), np.max(df_rgrp_gender_2['Rgrp']))

# Configurar a linha superior com dados do primeiro script
axs[0, 0].plot(df_rgrp_activity_1['Round'], df_rgrp_activity_1['Rgrp'], marker='o', color='blue')
axs[0, 0].set_title('Nível de Atividade (Script 1)')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[0, 0].set_ylim(y_min_activity, y_max_activity)

axs[0, 1].plot(df_rgrp_age_1['Round'], df_rgrp_age_1['Rgrp'], marker='o', color='blue')
axs[0, 1].set_title('Idade (Script 1)')
axs[0, 1].set_xlabel('Round')
axs[0, 1].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[0, 1].set_ylim(y_min_age, y_max_age)

axs[0, 2].plot(df_rgrp_gender_1['Round'], df_rgrp_gender_1['Rgrp'], marker='o', color='blue')
axs[0, 2].set_title('Gênero (Script 1)')
axs[0, 2].set_xlabel('Round')
axs[0, 2].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[0, 2].set_ylim(y_min_gender, y_max_gender)

# Configurar a linha inferior com dados do segundo script
axs[1, 0].plot(df_rgrp_activity_2['Round'], df_rgrp_activity_2['Rgrp'], marker='o', color='green')
axs[1, 0].set_title('Nível de Atividade (Script 2)')
axs[1, 0].set_xlabel('Round')
axs[1, 0].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[1, 0].set_ylim(y_min_activity, y_max_activity)

axs[1, 1].plot(df_rgrp_age_2['Round'], df_rgrp_age_2['Rgrp'], marker='o', color='green')
axs[1, 1].set_title('Idade (Script 2)')
axs[1, 1].set_xlabel('Round')
axs[1, 1].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[1, 1].set_ylim(y_min_age, y_max_age)

axs[1, 2].plot(df_rgrp_gender_2['Round'], df_rgrp_gender_2['Rgrp'], marker='o', color='green')
axs[1, 2].set_title('Gênero (Script 2)')
axs[1, 2].set_xlabel('Round')
axs[1, 2].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[1, 2].set_ylim(y_min_gender, y_max_gender)

# Ajustar layout e mostrar a figura
plt.tight_layout()
plt.show()
