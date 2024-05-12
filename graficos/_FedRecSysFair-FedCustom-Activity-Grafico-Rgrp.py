import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd

# Métricas por grupo de atividade
data_rgrp_activity = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.000587136, 0.000615752, 0.000647992, 0.000541797, 0.000279922, 4.5065e-05,
        2.5759e-05, 0.000224664, 0.000412887, 0.000608475, 0.000791455, 0.000906242,
        0.001140012, 0.001342053, 0.001488482, 0.001724044, 0.001940908, 0.002485204,
        0.003083606, 0.003613637, 0.003848011, 0.004577462, 0.005158144, 0.005699375, 0.005658691
    ]
}

data_rgrp_age = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.003316993264715157,
        0.003251807793377859,
        0.0032322931654841904,
        0.0035272411871037092,
        0.003970825756020964,
        0.004037469233907881,
        0.004009808075451978,
        0.003936471667819949,
        0.004081532651030693,
        0.003948377749837722,
        0.004491508798822626,
        0.004060774858755394,
        0.004578062433915229,
        0.0043192492932716655,
        0.004689115637215251,
        0.004699278789104403,
        0.005361180749725294,
        0.0062233703201010635,
        0.007333330523745089,
        0.008324128550660479,
        0.008918793010997456,
        0.01062044611971569,
        0.01171140963725474,
        0.01326773884178339,
        0.013325354506972096,
    ]
}

# Métricas por grupo de gênero
data_rgrp_gender = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        4.06835e-05, 0.000123035, 9.77093e-05, 0.000114679, 0.000199495, 0.000520983,
        0.001176677, 0.001961658, 0.002466697, 0.002917075, 0.003144436, 0.003394917,
        0.003508075, 0.003750874, 0.003827650, 0.004021368, 0.004059270, 0.004354577,
        0.004580626, 0.004821832, 0.004893485, 0.005136513, 0.005289502, 0.005387192, 0.005368775
    ]
}

# Criar DataFrames
df_rgrp_activity_losses = pd.DataFrame(data_rgrp_activity)
df_rgrp_age_losses = pd.DataFrame(data_rgrp_age)
df_rgrp_gender_losses = pd.DataFrame(data_rgrp_gender)


# Criar figura com 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Adicionar fundo cinza e quadrantes com bordas brancas para cada subplot
for ax in axs:
    ax.set_facecolor('#EAEAF2')  # Definindo a cor de fundo como cinza
    for spine in ax.spines.values():
        spine.set_visible(False)  # Removendo as bordas dos eixos
        spine.set_color('white')  # Definindo as bordas como brancas

# Adicionar os quadrantes no fundo dos gráficos
for ax in axs:
    ax.grid(color='white', linestyle='-.', linewidth=1)  # Adicionando os quadrantes com bordas brancas

# Plotar dados para "Favorecidos"
axs[0].plot(df_rgrp_activity_losses['Round'], df_rgrp_activity_losses['Rgrp'], marker='o', label='Rgrp', color='blue')
axs[0].set_title('Nível de Atividade')
axs[0].set_xlabel('Round')
axs[0].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[0].legend()

# Plotar dados para "Desfavorecidos"
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['Rgrp'], marker='o', label='Rgrp', color='blue')
axs[1].set_title('Idade')
axs[1].set_xlabel('Round')
axs[1].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[1].legend()

# Plotar dados para "Masculino" e "Feminino"
axs[2].plot(df_rgrp_gender_losses['Round'], df_rgrp_gender_losses['Rgrp'], marker='o', label='Rgrp', color='blue')
axs[2].set_title('Gênero')
axs[2].set_xlabel('Round')
axs[2].set_ylabel('Injustiça do Grupo (Rgrp)')
axs[2].legend()

plt.subplots_adjust(wspace=0.1)  # Reduz o espaço entre os gráficos
plt.tight_layout()
plt.show()
