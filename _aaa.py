data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02 = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.479313, 1.303436, 1.213227, 1.016362, 0.940068, 0.918444,
        0.907499, 0.899263, 0.897208, 0.894004, 0.891888, 0.890052,
        0.892762, 0.895690, 0.894634, 0.880448, 0.884265, 0.880780,
        0.875709, 0.874048, 0.878665, 0.880592, 0.887977, 0.889496,
        0.877194
    ],
    "Inativos": [
        1.394246, 1.257366, 1.180622, 1.020295, 0.965884, 0.949465,
        0.940800, 0.934871, 0.932327, 0.930570, 0.929666, 0.929275,
        0.930515, 0.932616, 0.932845, 0.928441, 0.932318, 0.931618,
        0.930944, 0.931648, 0.934079, 0.934896, 0.936914, 0.937520,
        0.931170
    ]
}

data_Losses_RgrpActivity_FedCustom_LossIndv_1 = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.479313, 1.303438, 1.213300, 1.016622, 0.941302, 0.918649, 
        0.906391, 0.903483, 0.897353, 0.894835, 0.898980, 0.897149, 
        0.893002, 0.895679, 0.895549, 0.878958, 0.871627, 0.866579, 
        0.872101, 0.868880, 0.872034, 0.862558, 0.864477, 0.865376, 
        0.863235
    ],
    "Inativos": [
        1.394246, 1.257367, 1.180683, 1.020347, 0.966467, 0.949525,
        0.940549, 0.936993, 0.932519, 0.931028, 0.932953, 0.932793,
        0.931204, 0.933973, 0.934984, 0.928862, 0.927076, 0.927933,
        0.930717, 0.931077, 0.932713, 0.931367, 0.932074, 0.932350,
        0.935160
    ]
}

data_Losses_RgrpActivity_FedAvg = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.479313, 1.273700, 1.168308, 0.985945, 0.922177, 0.896140, 
        0.881847, 0.869754, 0.865043, 0.859253, 0.856975, 0.852791, 
        0.852514, 0.845835, 0.841925, 0.842219, 0.839903, 0.839515, 
        0.839883, 0.838362, 0.837754, 0.836531, 0.835979, 0.835729, 
        0.836408
    ],
    "Inativos": [
        1.394246, 1.237118, 1.150059, 1.007524, 0.963306, 0.945759, 
        0.936582, 0.930227, 0.927241, 0.924946, 0.924239, 0.923482, 
        0.923680, 0.922368, 0.922121, 0.922580, 0.923107, 0.923738, 
        0.924508, 0.924435, 0.924488, 0.924283, 0.924269, 0.924248, 
        0.924025
    ]
}


# Função para calcular a média dos valores de "Ativos" e "Inativos"
def calcular_media(data):
    media_ativos = sum(data["Ativos"]) / len(data["Ativos"])
    media_inativos = sum(data["Inativos"]) / len(data["Inativos"])
    return media_ativos, media_inativos

# Dados fornecidos
data_sets = {
    "FedCustom_LossGroup_Activity_7_lambda02": data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02,
    "FedCustom_LossIndv_1": data_Losses_RgrpActivity_FedCustom_LossIndv_1,
    "FedAvg": data_Losses_RgrpActivity_FedAvg
}

# Calcular e imprimir as médias
for nome, data in data_sets.items():
    media_ativos, media_inativos = calcular_media(data)
    print(f"{nome} - Média Ativos: {media_ativos:.6f}, Média Inativos: {media_inativos:.6f}")
