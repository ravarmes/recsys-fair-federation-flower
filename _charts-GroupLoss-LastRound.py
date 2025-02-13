import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------------
# FedAvg(n)
# --------------------------------------------------------------------------------------------

data_losses_activity_FedAvgExample = {
    "Round": list(range(0, 25)),
    "Active": [
        1.479313, 1.273700, 1.168308, 0.985945, 0.922177, 0.896140, 
        0.881847, 0.869754, 0.865043, 0.859253, 0.856975, 0.852791, 
        0.852514, 0.845835, 0.841925, 0.842219, 0.839903, 0.839515, 
        0.839883, 0.838362, 0.837754, 0.836531, 0.835979, 0.835729, 
        0.836408
    ],
    "Inactive": [
        1.394246, 1.237118, 1.150059, 1.007524, 0.963306, 0.945759, 
        0.936582, 0.930227, 0.927241, 0.924946, 0.924239, 0.923482, 
        0.923680, 0.922368, 0.922121, 0.922580, 0.923107, 0.923738, 
        0.924508, 0.924435, 0.924488, 0.924283, 0.924269, 0.924248, 
        0.924025
    ]
}


data_losses_age_FedAvgExample = {
    "Round": list(range(0, 25)),
    "00-17": [
        1.468893, 1.316791, 1.211457, 1.045997, 0.988080, 0.962615, 
        0.947162, 0.934961, 0.928676, 0.924374, 0.922650, 0.922183, 
        0.921167, 0.917009, 0.914563, 0.912871, 0.913086, 0.913706, 
        0.913182, 0.914922, 0.915806, 0.916320, 0.917203, 0.918393, 
        0.918494
    ],
    "18-24": [
        1.441373, 1.290819, 1.201946, 1.054241, 1.002116, 0.979662,
        0.967006, 0.957766, 0.953031, 0.948892, 0.946793, 0.944448,
        0.943807, 0.940604, 0.939217, 0.939602, 0.939446, 0.939741,
        0.940614, 0.940105, 0.940177, 0.939871, 0.939644, 0.939245,
        0.938712
    ],
    "25-34": [
        1.425823, 1.256660, 1.164371, 1.007923, 0.957669, 0.937867, 
        0.927771, 0.920008, 0.916768, 0.913238, 0.911886, 0.909962, 
        0.909989, 0.907065, 0.905640, 0.906142, 0.905837, 0.906204, 
        0.906905, 0.906526, 0.906567, 0.906003, 0.905755, 0.905558, 
        0.905446
    ],
    "35-44": [
        1.327453, 1.175104, 1.094573, 0.969520, 0.934694, 0.920930,
        0.913628, 0.909286, 0.907040, 0.906485, 0.906890, 0.907628,
        0.908396, 0.909112, 0.910437, 0.910795, 0.912142, 0.912882,
        0.913726, 0.913899, 0.913744, 0.913822, 0.914095, 0.914402,
        0.914352
    ],
    "45-49": [
        1.316173, 1.163818, 1.083012, 0.953986, 0.918744, 0.905660, 
        0.898881, 0.895647, 0.894665, 0.895007, 0.896170, 0.897963, 
        0.899202, 0.900858, 0.902879 ,0.903876, 0.906032, 0.907783,
        0.908383, 0.908752, 0.909073, 0.909130, 0.909204, 0.909538,
        0.909406
    ],
    "50-55": [
        1.430696, 1.230131, 1.129084, 0.964145, 0.917408, 0.903098, 
        0.897045, 0.891657, 0.888770, 0.886016, 0.885442, 0.882620, 
        0.882963, 0.880452, 0.879058, 0.880866, 0.879398, 0.878987, 
        0.880348, 0.878026, 0.876461, 0.875432, 0.874971, 0.874379,
        0.874501
    ],
    "56-99": [
        1.417082, 1.321895, 1.257558, 1.174018, 1.154228, 1.145134, 
        1.139323, 1.140032, 1.138655, 1.143036, 1.145640, 1.152108,
        1.152537, 1.158133, 1.160980, 1.158273, 1.163176, 1.165518,
        1.165361, 1.169665, 1.170854, 1.173478, 1.175839, 1.177829,
        1.178019
    ]
}

data_losses_gender_FedAvgExample = {
    "Round": list(range(0, 25)),
    "M": [
        1.396886, 1.235448, 1.144062, 0.991679, 0.942558, 0.922970, 
        0.912748, 0.905407, 0.902244, 0.899388, 0.898436, 0.897275, 
        0.897444, 0.895274, 0.894520, 0.895026, 0.895221, 0.895791, 
        0.896499, 0.896343, 0.896485, 0.896184, 0.896081, 0.896054, 
        0.895942
    ],
    "F": [
        1.415898, 1.258143, 1.182107, 1.065392, 1.034138, 1.021848,
        1.015073, 1.010627, 1.007718, 1.006422, 1.006135, 1.005752,
        1.005887, 1.006097, 1.006557, 1.006761, 1.007580, 1.008073,
        1.008947, 1.008664, 1.008093, 1.007899, 1.008051, 1.007966,
        1.007625
    ]
}

data_losses_activity_FedAvgExample_goodbooks = {
    "Round": list(range(0, 25)),
    "Active": [
        2.018796, 1.412627, 1.275510, 1.209788, 1.159116, 1.108418, 
        1.052682, 1.010920, 0.983417, 0.963480, 0.951612, 0.951138, 
        0.950372, 0.949842, 0.952666, 0.957753, 0.956023, 0.949844, 
        0.945571, 0.941372, 0.939042, 0.938051, 0.937179, 0.933383, 
        0.931887
    ],
    "Inactive": [
        1.766799, 1.307901, 1.214422, 1.170626, 1.138357, 1.108201, 
        1.076092, 1.053373, 1.038681, 1.029187, 1.023865, 1.023172, 
        1.022733, 1.021966, 1.022560, 1.022966, 1.021526, 1.019569, 
        1.017715, 1.016274, 1.015868, 1.015495, 1.015595, 1.014878, 
        1.014773
    ]
}

# --------------------------------------------------------------------------------------------
# FedAvg(l)
# --------------------------------------------------------------------------------------------

data_losses_activity_FedAvgLoss = {
    "Round": list(range(25)),
    "Active": [
        1.479313, 1.303439, 1.213300, 1.016523, 0.940893, 0.918351,
        0.906299, 0.900292, 0.897871, 0.894385, 0.894083, 0.894288,
        0.886671, 0.895586, 0.886021, 0.889679, 0.884242, 0.889763,
        0.875707, 0.870150, 0.865792, 0.867510, 0.860547, 0.858772,
        0.857771
    ],
    "Inactive": [
        1.394246, 1.257367, 1.180683, 1.020265, 0.966297, 0.949614,
        0.940536, 0.935688, 0.932884, 0.930399, 0.930405, 0.931217,
        0.927754, 0.932283, 0.931953, 0.935869, 0.934475, 0.935834,
        0.932404, 0.931581, 0.931258, 0.933507, 0.932177, 0.931136,
        0.930286
    ]
}

data_losses_age_FedAvgLoss = {
    "Round": list(range(25)),
    "00-17": [
        1.468893, 1.336449, 1.242962, 1.056070, 0.991245, 0.965411,
        0.951456, 0.941898, 0.935907, 0.934597, 0.934074, 0.932503,
        0.930054, 0.923422, 0.922946, 0.910011, 0.913862, 0.921036,
        0.923476, 0.929614, 0.931567, 0.934448, 0.927527, 0.931847,
        0.942059
    ],
    "18-24": [
        1.441373, 1.310032, 1.232102, 1.066451, 1.005571, 0.981670,
        0.969170, 0.961279, 0.956560, 0.955286, 0.956025, 0.955114,
        0.952495, 0.952962, 0.952346, 0.947140, 0.942011, 0.947671,
        0.948033, 0.943990, 0.944317, 0.945738, 0.946446, 0.946506,
        0.950968
    ],
    "25-34": [
        1.425823, 1.278787, 1.197665, 1.022821, 0.963820, 0.942939,
        0.933641, 0.928551, 0.925714, 0.924096, 0.924400, 0.922185,
        0.921875, 0.921645, 0.923833, 0.919749, 0.916107, 0.918494,
        0.916165, 0.916307, 0.916479, 0.917336, 0.917479, 0.915269,
        0.919050
    ],
    "35-44": [
        1.327453, 1.194999, 1.123645, 0.981931, 0.939930, 0.925376,
        0.918119, 0.913704, 0.911638, 0.910349, 0.910811, 0.909228,
        0.910656, 0.914428, 0.914688, 0.915881, 0.916872, 0.917068,
        0.917231, 0.920277, 0.924920, 0.924126, 0.920754, 0.926163,
        0.926397
    ],
    "45-49": [
        1.316173, 1.183329, 1.112385, 0.965846, 0.924852, 0.910567,
        0.903624, 0.899853, 0.899240, 0.898252, 0.898771, 0.896606,
        0.897066, 0.896727, 0.896682, 0.899926, 0.904383, 0.904395,
        0.912025, 0.917803, 0.923183, 0.925386, 0.917676, 0.927863,
        0.928064
    ],
    "50-55": [
        1.430696, 1.259154, 1.173449, 0.992066, 0.936042, 0.920160,
        0.915126, 0.911011, 0.910264, 0.910142, 0.912248, 0.908181,
        0.908434, 0.911715, 0.906014, 0.901924, 0.892867, 0.892595,
        0.888868, 0.879958, 0.879489, 0.878002, 0.883326, 0.880367,
        0.878450
    ],
    "56-99": [
        1.417082, 1.331726, 1.275516, 1.178996, 1.158125, 1.150024,
        1.145671, 1.142368, 1.140679, 1.141527, 1.142151, 1.143702,
        1.145177, 1.148586, 1.151605, 1.159128, 1.163522, 1.148601,
        1.166924, 1.174388, 1.183647, 1.183524, 1.169718, 1.193422,
        1.183164
    ]
}

data_losses_gender_FedAvgLoss = {
    "Round": list(range(25)),
    "M": [
        1.396886, 1.256705, 1.176521, 1.006140, 0.946894, 0.928465,
        0.918228, 0.913037, 0.910167, 0.907417, 0.907245, 0.908391,
        0.904451, 0.909441, 0.907900, 0.912603, 0.911159, 0.912943,
        0.908166, 0.906493, 0.905937, 0.909320, 0.906197, 0.904565,
        0.903877
    ],
    "F": [
        1.415898, 1.277828, 1.210588, 1.077804, 1.037552, 1.025908,
        1.020528, 1.016667, 1.014287, 1.012521, 1.013152, 1.012338,
        1.009271, 1.013556, 1.014739, 1.015268, 1.012531, 1.013716,
        1.011826, 1.012736, 1.011836, 1.009146, 1.013139, 1.014281,
        1.012696
    ]
}

data_losses_activity_FedAvgLoss_goodbooks = {
    "Round": list(range(0, 25)),
    "Active": [
        2.444353, 1.469770, 1.340334, 1.296607, 1.245868, 1.208958, 
        1.140487, 1.134015, 1.110961, 1.093247, 1.082661, 1.089859, 
        1.098668, 1.114814, 1.136216, 1.152337, 1.169876, 1.157515, 
        1.183912, 1.174853, 1.140678, 1.125417, 1.149164, 1.132782, 
        1.165171
    ],
    "Inactive": [
        2.116478, 1.350972, 1.252687, 1.218733, 1.181936, 1.153368, 
        1.104193, 1.094740, 1.076866, 1.064537, 1.057967, 1.061733, 
        1.064673, 1.072860, 1.081837, 1.089491, 1.098485, 1.094873, 
        1.108624, 1.103354, 1.082981, 1.073194, 1.086260, 1.076527, 
        1.096652
    ]
}

# ----------------------------------------------------------
# FedFair(l)
# ----------------------------------------------------------
data_losses_activity_FedFairLoss = {
    "Round": list(range(0, 25)),
    "Active": [
        1.479313, 1.303436, 1.213227, 1.016362, 0.940068, 0.918444,
        0.907499, 0.899263, 0.897208, 0.894004, 0.891888, 0.890052,
        0.892762, 0.895690, 0.894634, 0.880448, 0.884265, 0.880780,
        0.875709, 0.874048, 0.878665, 0.880592, 0.887977, 0.889496,
        0.877194
    ],
    "Inactive": [
        1.394246, 1.257366, 1.180622, 1.020295, 0.965884, 0.949465,
        0.940800, 0.934871, 0.932327, 0.930570, 0.929666, 0.929275,
        0.930515, 0.932616, 0.932845, 0.928441, 0.932318, 0.931618,
        0.930944, 0.931648, 0.934079, 0.934896, 0.936914, 0.937520,
        0.931170
    ]
}

data_losses_age_FedFairLoss = {
    "Round": list(range(0, 25)),
    "00-17": [
        1.468893, 1.336215, 1.245376, 1.060750, 0.992381, 0.965537,
        0.952659, 0.943107, 0.937381, 0.936292, 0.933168, 0.933390,
        0.929656, 0.925753, 0.925305, 0.923476, 0.925601, 0.926765,
        0.924429, 0.936406, 0.930909, 0.923705, 0.920613, 0.916544,
        0.918259
    ],
    "18-24": [
        1.441373, 1.309798, 1.234643, 1.070155, 1.006698, 0.981740,
        0.969452, 0.961632, 0.956603, 0.955738, 0.954475, 0.958009,
        0.954749, 0.953950, 0.950482, 0.945568, 0.946460, 0.939626,
        0.943735, 0.946160, 0.948840, 0.950633, 0.958831, 0.948586,
        0.949653
    ],
    "25-34": [
        1.425823, 1.278522, 1.200455, 1.026498, 0.964498, 0.942734,
        0.933000, 0.928475, 0.925213, 0.923678, 0.921688, 0.924439,
        0.922427, 0.922599, 0.918885, 0.916025, 0.921148, 0.918566,
        0.923945, 0.922554, 0.920382, 0.921055, 0.927401, 0.919005,
        0.921915
    ],
    "35-44": [
        1.327453, 1.194762, 1.126050, 0.984777, 0.940849, 0.925995,
        0.918152, 0.913535, 0.911331, 0.909605, 0.909162, 0.910772,
        0.909021, 0.911420, 0.910289, 0.911242, 0.912947, 0.914675,
        0.914974, 0.915172, 0.917261, 0.917161, 0.920052, 0.921277,
        0.922881
    ],
    "45-49": [
        1.316173, 1.183092, 1.114974, 0.968702, 0.925208, 0.911180,
        0.904000, 0.899730, 0.898765, 0.896811, 0.896768, 0.897638,
        0.895549, 0.898039, 0.895681, 0.897709, 0.902800, 0.907537,
        0.909400, 0.913454, 0.916679, 0.912633, 0.911011, 0.914751,
        0.913346
    ],
    "50-55": [
        1.430696, 1.258807, 1.176648, 0.996408, 0.936833, 0.920677,
        0.913329, 0.908922, 0.908280, 0.907311, 0.906430, 0.911747,
        0.905873, 0.909461, 0.900316, 0.893703, 0.898144, 0.889659,
        0.894586, 0.881772, 0.881552, 0.886521, 0.905207, 0.897584,
        0.901206
    ],
    "56-99": [
        1.417082, 1.331603, 1.276474, 1.179345, 1.157719, 1.150700,
        1.147472, 1.145599, 1.142732, 1.143310, 1.146368, 1.143572,
        1.144282, 1.147708, 1.153103, 1.157998, 1.153782, 1.165031,
        1.144918, 1.148986, 1.149932, 1.143036, 1.127179, 1.143158,
        1.142802
    ]
}

data_losses_gender_FedFairLoss = {
    "Round": list(range(25)),
    "M": [
        1.396886, 1.256693, 1.176629, 1.007879, 0.947175, 0.927720,
        0.918949, 0.913085, 0.911266, 0.909114, 0.909524, 0.908765,
        0.910372, 0.909883, 0.906939, 0.906886, 0.905620, 0.904085,
        0.908360, 0.907575, 0.905214, 0.908161, 0.907278, 0.907214,
        0.921358
    ],
    "F": [
        1.415898, 1.277816, 1.210661, 1.079087, 1.038050, 1.026045,
        1.021364, 1.016630, 1.015027, 1.013505, 1.014002, 1.013498,
        1.016064, 1.015252, 1.011483, 1.007998, 1.006578, 1.006383,
        1.009055, 1.009180, 1.009621, 1.008587, 1.009130, 1.010321,
        1.012182
    ]
}

data_losses_activity_FedFairLoss_goodbooks = {
    "Round": list(range(0, 25)),
    "Active": [
        2.444353, 1.461461, 1.336614, 1.282655, 1.244086, 1.221441, 
        1.158010, 1.114219, 1.104571, 1.086639, 1.077236, 1.103736, 
        1.104704, 1.125071, 1.157166, 1.139251, 1.150389, 1.164589, 
        1.155895, 1.137512, 1.111916, 1.107646, 1.169994, 1.117005, 
        1.116592
    ],
    "Inactive": [
        2.116478, 1.345185, 1.250973, 1.209569, 1.181553, 1.160969, 
        1.115226, 1.084914, 1.075544, 1.063129, 1.057527, 1.070244, 
        1.069829, 1.080554, 1.095974, 1.083176, 1.088114, 1.096954, 
        1.093298, 1.082034, 1.069054, 1.068481, 1.109821, 1.079119, 
        1.082458
    ]
}

# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

age_labels = ["00-17", "18-24", "25-34", "35-44", "45-49", "50-55", "56-99"]

# Função auxiliar para calcular a média dos rounds
def round_mean(data, round_indices):
    return np.mean([data[i] for i in round_indices])

# Índices dos rounds 20, 21, 22, 23, 24
round_indices = [24]

# Função auxiliar para adicionar valores acima das barras com três casas decimais
def add_values_to_bars(ax, bars, decimal_places=2):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.{decimal_places}f}',
            ha='center',
            va='bottom'
        )

# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3, ax1_g), (ax4, ax5, ax6, ax4_g), (ax7, ax8, ax9, ax7_g)) = plt.subplots(3, 4, figsize=(15, 12))

# Aumenta o espaçamento entre as barras
bar_width = 0.4
bar_width_age = 0.4

# Subplot 1 - Atividade
bars1 = ax1.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedAvgExample["Active"], round_indices), 
                 round_mean(data_losses_activity_FedAvgExample["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax1.set_xticks(np.arange(2))
ax1.set_xticklabels(["Active", "Inactive"])
ax1.set_ylabel(r"FedAvg($n$)", fontsize=14)
ax1.set_title(r"MovieLens - Activity")
add_values_to_bars(ax1, bars1, decimal_places=3)
ax1.set_ylim(0, max([bar.get_height() for bar in bars1]) * 1.2)  # Ajusta a escala y

# Subplot 2 - Idade
bars2 = []
for idx, label in enumerate(age_labels):
    bars2.extend(ax2.bar(idx, 
                         round_mean(data_losses_age_FedAvgExample[label], round_indices), 
                         width=bar_width_age, label=label))
ax2.set_xticks(np.arange(len(age_labels)))
ax2.set_xticklabels(age_labels)
ax2.set_title(r"MovieLens - Age")
add_values_to_bars(ax2, bars2)
ax2.set_ylim(0, max([bar.get_height() for bar in bars2]) * 1.2)  # Ajusta a escala y

# Subplot 3 - Gênero
bars3 = ax3.bar(np.arange(2), 
                [round_mean(data_losses_gender_FedAvgExample["M"], round_indices), 
                 round_mean(data_losses_gender_FedAvgExample["F"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax3.set_xticks(np.arange(2))
ax3.set_xticklabels(["Male", "Female"])
ax3.set_title(r"MovieLens - Gender")
add_values_to_bars(ax3, bars3, decimal_places=3)
ax3.set_ylim(0, max([bar.get_height() for bar in bars3]) * 1.2)  # Ajusta a escala y

# Subplot 4 - Atividade
bars4 = ax4.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedAvgLoss["Active"], round_indices), 
                 round_mean(data_losses_activity_FedAvgLoss["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax4.set_xticks(np.arange(2))
ax4.set_xticklabels(["Active", "Inactive"])
ax4.set_ylabel(r"FedAvg($\ell$)", fontsize=14)
add_values_to_bars(ax4, bars4, decimal_places=3)
ax4.set_ylim(0, max([bar.get_height() for bar in bars4]) * 1.2)  # Ajusta a escala y

# Subplot 5 - Idade
bars5 = []
for idx, label in enumerate(age_labels):
    bars5.extend(ax5.bar(idx, 
                         round_mean(data_losses_age_FedAvgLoss[label], round_indices), 
                         width=bar_width_age, label=label))
ax5.set_xticks(np.arange(len(age_labels)))
ax5.set_xticklabels(age_labels)
add_values_to_bars(ax5, bars5)
ax5.set_ylim(0, max([bar.get_height() for bar in bars5]) * 1.2)  # Ajusta a escala y

# Subplot 6 - Gênero
bars6 = ax6.bar(np.arange(2), 
                [round_mean(data_losses_gender_FedAvgLoss["M"], round_indices), 
                 round_mean(data_losses_gender_FedAvgLoss["F"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax6.set_xticks(np.arange(2))
ax6.set_xticklabels(["Male", "Female"])
add_values_to_bars(ax6, bars6, decimal_places=3)
ax6.set_ylim(0, max([bar.get_height() for bar in bars6]) * 1.2)  # Ajusta a escala y

# Subplot 7 - Atividade
bars7 = ax7.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedFairLoss["Active"], round_indices), 
                 round_mean(data_losses_activity_FedFairLoss["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax7.set_xticks(np.arange(2))
ax7.set_xticklabels(["Active", "Inactive"])
ax7.set_ylabel(r"FedFair$(\ell)$", fontsize=14)
add_values_to_bars(ax7, bars7, decimal_places=3)
ax7.set_ylim(0, max([bar.get_height() for bar in bars7]) * 1.2)  # Ajusta a escala y

# Subplot 8 - Idade
bars8 = []
for idx, label in enumerate(age_labels):
    bars8.extend(ax8.bar(idx, 
                         round_mean(data_losses_age_FedFairLoss[label], round_indices), 
                         width=bar_width_age, label=label))
ax8.set_xticks(np.arange(len(age_labels)))
ax8.set_xticklabels(age_labels)
add_values_to_bars(ax8, bars8)
#ax8.set_ylim(0, max([bar.get_height() for bar in bars8]) * 1.2)  # Ajusta a escala y
ax8.set_ylim(0, 1.416)

# Subplot 9 - Gênero
bars9 = ax9.bar(np.arange(2), 
                [round_mean(data_losses_gender_FedFairLoss["M"], round_indices), 
                 round_mean(data_losses_gender_FedFairLoss["F"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax9.set_xticks(np.arange(2))
ax9.set_xticklabels(["Male", "Female"])
add_values_to_bars(ax9, bars9, decimal_places=3)
ax9.set_ylim(0, max([bar.get_height() for bar in bars9]) * 1.2)  # Ajusta a escala y

# --------------------------------------------------------------------------------
# GOODBOOKS
# --------------------------------------------------------------------------------
bars1 = ax1_g.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedAvgExample_goodbooks["Active"], round_indices), 
                 round_mean(data_losses_activity_FedAvgExample_goodbooks["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax1_g.set_xticks(np.arange(2))
ax1_g.set_xticklabels(["Active", "Inactive"])
ax1_g.set_title(r"GoodBooks - Activity")
add_values_to_bars(ax1_g, bars1, decimal_places=3)
ax1_g.set_ylim(0, max([bar.get_height() for bar in bars1]) * 1.2)  # Ajusta a escala y

bars4 = ax4_g.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedAvgLoss_goodbooks["Active"], round_indices), 
                 round_mean(data_losses_activity_FedAvgLoss_goodbooks["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax4_g.set_xticks(np.arange(2))
ax4_g.set_xticklabels(["Active", "Inactive"])
add_values_to_bars(ax4_g, bars4, decimal_places=3)
ax4_g.set_ylim(0, max([bar.get_height() for bar in bars4]) * 1.2)  # Ajusta a escala y

bars7 = ax7_g.bar(np.arange(2), 
                [round_mean(data_losses_activity_FedFairLoss_goodbooks["Active"], round_indices), 
                 round_mean(data_losses_activity_FedFairLoss_goodbooks["Inactive"], round_indices)],
                width=bar_width, color=['blue', 'orange'])  # Especifica cores diferentes
ax7_g.set_xticks(np.arange(2))
ax7_g.set_xticklabels(["Active", "Inactive"])
add_values_to_bars(ax7_g, bars7, decimal_places=3)
ax7_g.set_ylim(0, max([bar.get_height() for bar in bars7]) * 1.2)  # Ajusta a escala y

# Ajustar espaçamento entre subplots
fig.suptitle("Group Average Loss", fontsize=16)
plt.subplots_adjust(hspace=0.8, wspace=0.5)
plt.tight_layout()
plt.show()
