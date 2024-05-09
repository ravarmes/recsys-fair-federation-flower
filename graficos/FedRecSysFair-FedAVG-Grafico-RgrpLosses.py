import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd

# Dicionários de saída
RgrpActivity_Losses = {
    "Round": list(range(25)),
    "Favorecidos": [1.429528, 1.232358, 1.355108, 1.323403, 1.273502, 1.167273, 1.076172, 1.020529, 1.001100,
                    0.994006, 1.000201, 1.005862, 1.022481, 1.030620, 1.041488, 1.031717, 1.017279, 1.007053,
                    0.994729, 0.987076, 0.983368, 0.980308, 0.972611, 0.968000, 0.971199],
    "Desfavorecidos": [1.366547, 1.209986, 1.301725, 1.276256, 1.237340, 1.151692, 1.081070, 1.042611, 1.031163,
                       1.030717, 1.040590, 1.050265, 1.070186, 1.082268, 1.095978, 1.094386, 1.086482, 1.083880,
                       1.081103, 1.082549, 1.086923, 1.090168, 1.091915, 1.095825, 1.106508]
}

RgrpAge_Losses = {
    "Round": list(range(25)),
    "00-17": [1.413159, 1.286538, 1.370368, 1.345130, 1.305859, 1.210157, 1.125469, 1.076360, 1.050892,
              1.036493, 1.031008, 1.026619, 1.030365, 1.031463, 1.034771, 1.028379, 1.017071, 1.011344,
              1.006219, 1.006861, 1.009675, 1.011698, 1.015458, 1.020898, 1.031175],
    "18-24": [1.424959, 1.270873, 1.355012, 1.331000, 1.293681, 1.209624, 1.136257, 1.092295, 1.076785,
              1.074864, 1.085216, 1.093911, 1.113370, 1.125771, 1.138002, 1.134114, 1.122340, 1.116369,
              1.108726, 1.106409, 1.105389, 1.104339, 1.100127, 1.098163, 1.106098],
    "25-34": [1.396978, 1.226936, 1.329148, 1.301626, 1.259825, 1.166907, 1.087913, 1.042961, 1.029907,
              1.028455, 1.038976, 1.047932, 1.068029, 1.079288, 1.092265, 1.087412, 1.076600, 1.070415,
              1.063421, 1.060396, 1.061138, 1.061133, 1.059297, 1.059851, 1.068175],
    "35-44": [1.293850, 1.149902, 1.233005, 1.209583, 1.172312, 1.093639, 1.032804, 1.001499, 0.992790,
              0.994130, 1.003765, 1.015835, 1.036769, 1.050525, 1.066959, 1.069603, 1.066220, 1.068973,
              1.072171, 1.079412, 1.089055, 1.097346, 1.104180, 1.112971, 1.126096],
    "45-49": [1.294633, 1.138089, 1.223279, 1.198337, 1.162467, 1.082411, 1.021468, 0.991217, 0.981752,
              0.978357, 0.983882, 0.991189, 1.008207, 1.018600, 1.031644, 1.035410, 1.033724, 1.037153,
              1.042074, 1.052445, 1.065562, 1.075970, 1.086514, 1.099837, 1.116052],
    "50-55": [1.358467, 1.148843, 1.298611, 1.265175, 1.217956, 1.121220, 1.048643, 1.017710, 1.015744,
              1.023991, 1.039563, 1.056230, 1.081976, 1.098093, 1.114303, 1.112747, 1.100785, 1.097141,
              1.091591, 1.088695, 1.090146, 1.089375, 1.084049, 1.080522, 1.085421],
    "56-99": [1.378431, 1.324048, 1.354798, 1.334811, 1.296644, 1.229898, 1.183001, 1.171498, 1.161019,
              1.161228, 1.159232, 1.162878, 1.173635, 1.178792, 1.191086, 1.195578, 1.206116, 1.215588,
              1.235774, 1.256658, 1.282586, 1.305908, 1.334826, 1.363730, 1.391056]
}

RgrpGender_Losses = {
    "Round": list(range(25)),
    "Masculino": [1.368577, 1.206919, 1.301823, 1.275559, 1.234688, 1.144016, 1.067038, 1.023044,
                  1.008868, 1.006407, 1.015059, 1.023156, 1.042120, 1.053108, 1.066141, 1.063245,
                  1.054191, 1.050581, 1.046535, 1.046651, 1.049712, 1.052107, 1.052695, 1.055571,
                  1.065370],
    "Feminino": [1.382260, 1.231385, 1.321818, 1.297275, 1.262302, 1.189727, 1.137778, 1.115825,
                 1.112704, 1.118115, 1.131677, 1.146396, 1.169045, 1.184180, 1.199627, 1.200336,
                 1.194726, 1.193403, 1.192257, 1.195762, 1.202516, 1.206884, 1.209844, 1.214799,
                 1.226294]
}

# Criar DataFrames
df_rgrp_activity_losses = pd.DataFrame(RgrpActivity_Losses)
df_rgrp_age_losses = pd.DataFrame(RgrpAge_Losses)
df_rgrp_gender_losses = pd.DataFrame(RgrpGender_Losses)


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
