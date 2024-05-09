import matplotlib.pyplot as plt
import pandas as pd

# Dados fornecidos
RgrpActivity_Losses = {
    "Round": list(range(0, 25)),
    "Favorecidos": [
        1.428404, 1.292772, 1.324682, 1.286520, 1.209032,
        1.114784, 1.036564, 0.988784, 0.964168, 0.951584,
        0.946421, 0.946657, 0.940464, 0.941457, 0.941236,
        0.943307, 0.937967, 0.927507, 0.920201, 0.919385,
        0.922105, 0.918488, 0.912963, 0.912137, 0.909217
    ],
    "Desfavorecidos": [
        1.379942, 1.243143, 1.273771, 1.239967, 1.175570,
        1.101358, 1.046714, 1.018761, 1.004807, 1.000918,
        1.002687, 1.006865, 1.007992, 1.014726, 1.018398,
        1.026350, 1.026078, 1.027211, 1.031261, 1.039612,
        1.046169, 1.053802, 1.056603, 1.063125, 1.059665
    ]
}

data_rgrp_age_losses = {
    "Round": list(range(25)),
    "00-17": [1.462244, 1.329887, 1.355081, 1.327405, 1.267247, 1.186650, 1.113167,
    1.062811, 1.029456, 1.008599, 0.995558, 0.988606, 0.980984, 0.979968,
    0.977557, 0.981021, 0.976916, 0.976567, 0.982267, 0.991083, 0.997679,
    1.005860, 1.011208, 1.018371, 1.016343],
    "18-24": [1.423006, 1.296675, 1.326471, 1.293631, 1.230596, 1.155810, 1.097022,
    1.064181, 1.048538, 1.043114, 1.042629, 1.045843, 1.045194, 1.051108,
    1.053680, 1.060421, 1.058144, 1.054142, 1.053609, 1.057192, 1.061558,
    1.062796, 1.061537, 1.064323, 1.059997],
    "25-34": [1.411283, 1.265573, 1.298726, 1.261683, 1.190892, 1.108039, 1.045417,
    1.012598, 0.996823, 0.992141, 0.993136, 0.997591, 0.996969, 1.002698,
    1.004984, 1.011031, 1.008504, 1.006162, 1.005904, 1.010561, 1.014810,
    1.018466, 1.018283, 1.022091, 1.018126],
    "35-44": [1.306542, 1.184010, 1.208047, 1.177360, 1.119045, 1.053743, 1.008746,
    0.987457, 0.976186, 0.973995, 0.977416, 0.982499, 0.985562, 0.994228,
    0.999460, 1.009689, 1.011940, 1.017209, 1.025730, 1.039022, 1.048202,
    1.060656, 1.066788, 1.076470, 1.073939],
    "45-49": [1.296056, 1.168541, 1.197827, 1.166401, 1.107340, 1.043269, 0.998070,
    0.974594, 0.960749, 0.954487, 0.955805, 0.958520, 0.961949, 0.968900,
    0.974794, 0.986028, 0.990961, 0.999291, 1.013279, 1.029756, 1.041674,
    1.059902, 1.070259, 1.083539, 1.082122],
    "50-55": [1.406654, 1.225719, 1.270634, 1.231359, 1.156619, 1.076827, 1.020821,
    0.996075, 0.981987, 0.982577, 0.983306, 0.989755, 0.989309, 0.997340,
    1.001339, 1.009693, 1.006651, 1.005757, 1.005467, 1.011168, 1.018452,
    1.021761, 1.021453, 1.023353, 1.020623],
    "56-99": [1.410409, 1.299353, 1.330688, 1.314149, 1.275080, 1.221280, 1.186553,
    1.169440, 1.162450, 1.156051, 1.169669, 1.162131, 1.174485, 1.173914,
    1.185271, 1.193388, 1.207267, 1.226683, 1.252204, 1.277554, 1.294000,
    1.326164, 1.343740, 1.369127, 1.366879]
}

data_rgrp_gender_losses = {
    "Round": list(range(0, 25)),  # Lista de 0 a 24
    "Masculino": [
        1.381074, 1.242540, 1.273729, 1.239283, 1.172595,
        1.093534, 1.032705, 0.999420, 0.982597, 0.976385,
        0.976840, 0.979880, 0.980091, 0.985620, 0.988762,
        0.995680, 0.994917, 0.994309, 0.996863, 1.003854,
        1.009928, 1.016062, 1.017836, 1.023554, 1.020183,
    ],
    "Feminino": [
        1.393830, 1.264724, 1.293499, 1.260701, 1.200844,
        1.139184, 1.101310, 1.088001, 1.081929, 1.084404,
        1.088991, 1.096411, 1.098549, 1.108108, 1.112498,
        1.122509, 1.122342, 1.126288, 1.132224, 1.142733,
        1.149835, 1.159401, 1.163294, 1.170349, 1.166727,
    ]
}

# Criando o DataFrame
df_rgrp_activity_losses = pd.DataFrame(RgrpActivity_Losses)
df_rgrp_age_losses = pd.DataFrame(data_rgrp_age_losses)
df_rgrp_gender_losses = pd.DataFrame(data_rgrp_gender_losses)

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
axs[0].plot(df_rgrp_activity_losses['Round'], df_rgrp_activity_losses['Favorecidos'], marker='o', label='Favorecidos', color='blue')
axs[0].plot(df_rgrp_activity_losses['Round'], df_rgrp_activity_losses['Desfavorecidos'], marker='o', label='Desfavorecidos', color='green')
axs[0].set_title('Nível de Atividade')
axs[0].set_xlabel('Round')
axs[0].set_ylabel('Injustiça do Grupo')
axs[0].legend()

# Plotar dados para "Desfavorecidos"
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['00-17'], marker='o', label='00-17', color='blue')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['18-24'], marker='o', label='18-24', color='green')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['25-34'], marker='o', label='25-34', color='red')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['35-44'], marker='o', label='35-44', color='gray')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['45-49'], marker='o', label='45-49', color='orange')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['50-55'], marker='o', label='50-55', color='brown')
axs[1].plot(df_rgrp_activity_losses['Round'], df_rgrp_age_losses['56-99'], marker='o', label='56-99', color='purple')
axs[1].set_title('Idade')
axs[1].set_xlabel('Round')
axs[1].set_ylabel('Injustiça do Grupo')
axs[1].legend()

# Plotar dados para "Masculino" e "Feminino"
axs[2].plot(df_rgrp_gender_losses['Round'], df_rgrp_gender_losses['Masculino'], marker='o', label='Masculino', color='blue')
axs[2].plot(df_rgrp_gender_losses['Round'], df_rgrp_gender_losses['Feminino'], marker='o', label='Feminino', color='green')
axs[2].set_title('Gênero')
axs[2].set_xlabel('Round')
axs[2].set_ylabel('Injustiça do Grupo')
axs[2].legend()

plt.subplots_adjust(wspace=0.1)  # Reduz o espaço entre os gráficos
plt.tight_layout()
plt.show()
