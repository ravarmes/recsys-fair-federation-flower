import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd

# Métricas por grupo de atividade
data_rgrp_activity = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.000981994, 0.000238225, 0.000504384, 0.000430394, 0.000188408, 3.596e-06, 
        0.000134395, 0.000476567, 0.000709533, 0.000955670, 0.001092158, 0.001086757, 
        0.001186913, 0.001348532, 0.001563960, 0.001856400, 0.002241598, 0.002600655, 
        0.003121024, 0.003715837, 0.004516703, 0.005286370, 0.006090524, 0.006575472, 
        0.007342683
    ]
}

data_rgrp_age = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        0.003396680190163166,
        0.003883012178654958,
        0.0032389046429972,
        0.0033415192432913977,
        0.003622837286258953,
        0.004139421103580343,
        0.004522999975863927,
        0.004788569180378254,
        0.004456227536866096,
        0.004230801833006068,
        0.0039457243582087425,
        0.003498848430346345,
        0.003322188539265435,
        0.0033420332522783544,
        0.0036406929257304195,
        0.003894552862944088,
        0.004579852897645133,
        0.005006362570400268,
        0.005874461016865226,
        0.006780175495069189,
        0.00815280980914856,
        0.00949522482741473,
        0.010576392752614999,
        0.01133786831348063,
        0.01261870780876268
    ]

}

# Métricas por grupo de gênero
data_rgrp_gender = {
    "Round": list(range(0, 25)),
    "Rgrp": [
        9.39706e-06, 0.000111572, 9.23792e-05, 0.000143523, 0.000352234, 0.000953561, 
        0.00190807, 0.00279572, 0.00327143, 0.00362377, 0.00381965, 0.00389468, 
        0.00402243, 0.00412658, 0.00431023, 0.00446346, 0.00467865, 0.00481244, 
        0.00503336, 0.00527175, 0.00548703, 0.00571105, 0.00594851, 0.00605597, 
        0.00620417
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
