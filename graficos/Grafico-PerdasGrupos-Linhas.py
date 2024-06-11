import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# FedAvg
# --------------------------------------------------------------------------------------------

data_Losses_RgrpActivity_FedAvg = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.432424, 1.244771, 1.293405, 1.265230, 1.185315, 1.087239, 
        1.013768, 0.965486, 0.941803, 0.932693, 0.931094, 0.939602, 
        0.946563, 0.950219, 0.951937, 0.945578, 0.937862, 0.928369, 
        0.918190, 0.910171, 0.899080, 0.893008, 0.894234, 0.892460, 
        0.894878
    ],
    "Inativos": [
        1.369751, 1.213902, 1.248488, 1.223738, 1.157863, 1.083446, 
        1.036954, 1.009147, 0.995077, 0.994520, 0.997190, 1.005534, 
        1.015466, 1.023664, 1.031030, 1.031750, 1.032553, 1.030363, 
        1.029922, 1.032086, 1.033493, 1.038423, 1.050318, 1.054638, 
        1.066257
    ]
}

data_Losses_RgrpAge_FedAvg = {
    "Round": list(range(0, 25)),
    "00-17": [
        1.444010, 1.300652, 1.326861, 1.302489, 1.237340, 1.157655, 
        1.095042, 1.049193, 1.017078, 1.001018, 0.992476, 0.989838, 
        0.987796, 0.988692, 0.986884, 0.985004, 0.983020, 0.980465, 
        0.979803, 0.982075, 0.986600, 0.993027, 1.005559, 1.010593, 
        1.024532
    ],
    "18-24": [
        1.429315, 1.265223, 1.299307, 1.275173, 1.207711, 1.128787, 
        1.074742, 1.041118, 1.024767, 1.023577, 1.025953, 1.034965, 
        1.044574, 1.053481, 1.059053, 1.057716, 1.054780, 1.050120, 
        1.045166, 1.043307, 1.039581, 1.039725, 1.047532, 1.048979, 
        1.056041
    ],
    "25-34": [
        1.399730, 1.232681, 1.271766, 1.245575, 1.175302, 1.092003, 
        1.036197, 1.001107, 0.984453, 0.982636, 0.985138, 0.994752, 
        1.004558, 1.012209, 1.017960, 1.016180, 1.014062, 1.009495, 
        1.005653, 1.004085, 1.001701, 1.002967, 1.011048, 1.013123, 
        1.021808
    ],
    "35-44": [
        1.295373, 1.154616, 1.185257, 1.162134, 1.102370, 1.040576, 
        1.006890, 0.988574, 0.977808, 0.978188, 0.980545, 0.987651, 
        0.997901, 1.005934, 1.014806, 1.018372, 1.022842, 1.023332, 
        1.026973, 1.033023, 1.038232, 1.046551, 1.062408, 1.069109, 
        1.084415
    ],
    "45-49": [
        1.284259, 1.146454, 1.177151, 1.152171, 1.089611, 1.023345, 
        0.988832, 0.971097, 0.959367, 0.957992, 0.959810, 0.964481, 
        0.974027, 0.981637, 0.992996, 0.998609, 1.005240, 1.007848, 
        1.015077, 1.026275, 1.036799, 1.051855, 1.072982, 1.083178, 
        1.102128
    ],
    "50-55": [
        1.376054, 1.181862, 1.234692, 1.207273, 1.127671, 1.040065, 
        0.990049, 0.965726, 0.961055, 0.967452, 0.976605, 0.991936, 
        1.006667, 1.015811, 1.024464, 1.023769, 1.022257, 1.017345, 
        1.013636, 1.012072, 1.006923, 1.007279, 1.015985, 1.018909, 
        1.025792
    ],
    "56-99": [
        1.404795, 1.307614, 1.318594, 1.299586, 1.250914, 1.208497, 
        1.190661, 1.181316, 1.166004, 1.161907, 1.158153, 1.153708, 
        1.156918, 1.163107, 1.176880, 1.184617, 1.201821, 1.209377, 
        1.226985, 1.245975, 1.270388, 1.295223, 1.321462, 1.335551, 
        1.362505
    ]
}

data_Losses_RgrpGender_FedAvg = {
    "Round": list(range(0, 25)),
    "M": [
        1.373217, 1.212111, 1.248107, 1.222188, 1.152647, 1.071794, 
        1.018352, 0.985473, 0.969019, 0.966668, 0.968400, 0.976523, 
        0.985842, 0.993388, 0.999789, 0.999536, 0.999092, 0.995984, 
        0.994212, 0.994979, 0.994891, 0.998425, 1.008927, 1.012528, 
        1.023097
    ],
    "F": [
        1.379348, 1.233237, 1.267330, 1.246148, 1.190183, 1.133554, 
        1.105715, 1.091222, 1.083412, 1.087064, 1.092007, 1.101338, 
        1.112687, 1.121865, 1.131093, 1.133154, 1.135893, 1.134727, 
        1.136105, 1.140193, 1.143040, 1.149568, 1.163181, 1.168168, 
        1.180630
    ]
}

# --------------------------------------------------------------------------------------------
# Fed(l)
# --------------------------------------------------------------------------------------------

data_Losses_RgrpActivity_FedCustom_LossIndv_1 = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.403444, 1.302421, 1.363693, 1.357039, 1.301333, 1.187420, 
        1.114929, 1.055727, 1.029991, 1.023993, 1.033316, 1.044845, 
        1.047135, 1.045598, 1.043032, 1.036002, 1.020798, 1.018890, 
        1.018102, 1.002819, 0.995377, 0.997582, 0.996874, 1.003763, 
        0.976214
    ],
    "Inativos": [
        1.335036, 1.257038, 1.306468, 1.299654, 1.248514, 1.147708, 
        1.085630, 1.043919, 1.030592, 1.033370, 1.045415, 1.061409, 
        1.071365, 1.077132, 1.078535, 1.078225, 1.077663, 1.083006, 
        1.091809, 1.087810, 1.094890, 1.102191, 1.107293, 1.127185, 
        1.091921
    ]
}

data_Losses_RgrpAge_FedCustom_LossIndv_1 = {
    "Round": list(range(0, 25)),
    "00-17": [
        1.422988, 1.325416, 1.378673, 1.368607, 1.309311, 1.194752, 
        1.114009, 1.060192, 1.033854, 1.022630, 1.022831, 1.027593, 
        1.027442, 1.022405, 1.022585, 1.013240, 1.005359, 1.005240, 
        1.009606, 1.008589, 1.016694, 1.021452, 1.036510, 1.052521, 
        1.027086
    ],
    "18-24": [
        1.377059, 1.311517, 1.358052, 1.350989, 1.300555, 1.195387, 
        1.127943, 1.080090, 1.063267, 1.063819, 1.076870, 1.094329, 
        1.104180, 1.111193, 1.109977, 1.108078, 1.104983, 1.109224, 
        1.116621, 1.107372, 1.109963, 1.113462, 1.114674, 1.129051, 
        1.099385
    ],
    "25-34": [
        1.368172, 1.281434, 1.334907, 1.327896, 1.273668, 1.165045, 
        1.095495, 1.046536, 1.029600, 1.030912, 1.045174, 1.061316, 
        1.069643, 1.074528, 1.074755, 1.071935, 1.066640, 1.069892, 
        1.076730, 1.067815, 1.070378, 1.074495, 1.076146, 1.091653, 
        1.059698
    ],
    "35-44": [
        1.264338, 1.192679, 1.237095, 1.231467, 1.185516, 1.096556, 
        1.045929, 1.014186, 1.005989, 1.011810, 1.022177, 1.037673, 
        1.049536, 1.055868, 1.059177, 1.062069, 1.065027, 1.073208, 
        1.082903, 1.084232, 1.095895, 1.106782, 1.117058, 1.143054, 
        1.103705
    ],
    "45-49": [
        1.255152, 1.181801, 1.228591, 1.221412, 1.172533, 1.082108, 
        1.030200, 0.995925, 0.984098, 0.985648, 0.989298, 1.000944, 
        1.011016, 1.015401, 1.020523, 1.023996, 1.029661, 1.031893, 
        1.043934, 1.052036, 1.069226, 1.085002, 1.100326, 1.126210, 
        1.088166
    ],
    "50-55": [
        1.355162, 1.232800, 1.309168, 1.301249, 1.238029, 1.129998, 
        1.072498, 1.034188, 1.026563, 1.031787, 1.047281, 1.067483, 
        1.076565, 1.077772, 1.077445, 1.076744, 1.074940, 1.080525, 
        1.084518, 1.074841, 1.076044, 1.087135, 1.086326, 1.099041, 
        1.066692
    ],
    "56-99": [
        1.371159, 1.326373, 1.354495, 1.349548, 1.317791, 1.256076, 
        1.229569, 1.221478, 1.220071, 1.225490, 1.219897, 1.226559, 
        1.237806, 1.244086, 1.251350, 1.259913, 1.287251, 1.313875, 
        1.336839, 1.358964, 1.381219, 1.402237, 1.410478, 1.466139, 
        1.368923
    ]
}

data_Losses_RgrpGender_FedCustom_LossIndv_1 = {
    "Round": list(range(0, 25)),
    "M": [
        1.335223, 1.256127, 1.306750, 1.299949, 1.247112, 1.140222, 
        1.072485, 1.025562, 1.008581, 1.008692, 1.020318, 1.035366, 
        1.044353, 1.049778, 1.050855, 1.049603, 1.047517, 1.052006, 
        1.060224, 1.054579, 1.060363, 1.066026, 1.070399, 1.088393, 
        1.053590
    ],
    "F": [
        1.360533, 1.278275, 1.327273, 1.320463, 1.274659, 1.194218, 
        1.151768, 1.125098, 1.122263, 1.132807, 1.145554, 1.163780, 
        1.174842, 1.179233, 1.180471, 1.181513, 1.181685, 1.187811, 
        1.195374, 1.193915, 1.200826, 1.213009, 1.218922, 1.241747, 
        1.207518
    ]
}

# ----------------------------------------------------------
# FairFed(\lambda\ell)
# ----------------------------------------------------------
data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02 = {
    "Round": list(range(0, 25)),
    "Ativos": [
        1.505397, 1.509568, 1.550259, 1.506446, 1.445334, 1.257832, 
        1.167460, 1.141375, 1.137118, 1.157180, 1.199587, 1.209470, 
        1.311123, 1.227259, 1.206824, 1.297942, 1.177427, 1.150979, 
        1.160952, 1.156057, 1.157347, 1.130909, 1.104549, 1.099209, 
        1.092578
    ],
    "Inativos": [
        1.448849, 1.433882, 1.467120, 1.424714, 1.368906, 1.199774, 
        1.125686, 1.106523, 1.109878, 1.128228, 1.175305, 1.187044, 
        1.260998, 1.199758, 1.201810, 1.258902, 1.183026, 1.169809, 
        1.178295, 1.175701, 1.178919, 1.155998, 1.133556, 1.129225, 
        1.126974
    ]
}

data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02 = {
    "Round": list(range(0, 25)),
    "00-17": [
        1.490926, 1.503190, 1.537215, 1.493195, 1.431071, 1.252757, 
        1.153348, 1.116935, 1.097732, 1.099799, 1.126813, 1.126938, 
        1.200561, 1.142862, 1.131887, 1.201009, 1.127625, 1.113360, 
        1.117632, 1.107753, 1.111029, 1.092242, 1.064789, 1.053392, 
        1.047421
    ],
    "18-24": [
        1.484975, 1.478842, 1.513798, 1.473409, 1.416327, 1.249492, 
        1.171512, 1.152888, 1.155958, 1.177352, 1.229992, 1.244126, 
        1.329473, 1.265184, 1.263889, 1.336517, 1.250857, 1.232513, 
        1.240860, 1.233571, 1.241447, 1.211232, 1.182291, 1.178300, 
        1.174541
    ],
    "25-34": [
        1.487800, 1.468531, 1.503678, 1.458660, 1.399817, 1.217222, 
        1.133402, 1.111269, 1.112962, 1.132003, 1.180890, 1.193387, 
        1.278600, 1.208215, 1.203297, 1.272832, 1.180177, 1.160977, 
        1.168640, 1.167674, 1.172983, 1.146453, 1.122336, 1.119578, 
        1.116950
    ],
    "35-44": [
        1.364083, 1.361838, 1.391262, 1.352380, 1.300251, 1.148481, 
        1.087239, 1.073802, 1.079987, 1.100120, 1.143366, 1.154349, 
        1.216330, 1.163796, 1.173270, 1.216971, 1.158393, 1.151349, 
        1.159201, 1.156265, 1.152807, 1.135826, 1.116652, 1.109313, 
        1.108502
    ],
    "45-49": [
        1.369900, 1.352137, 1.384499, 1.344899, 1.295164, 1.136674, 
        1.071431, 1.051691, 1.054404, 1.067072, 1.105987, 1.112049, 
        1.157774, 1.111130, 1.121576, 1.145083, 1.097983, 1.097286, 
        1.109499, 1.110435, 1.107737, 1.096246, 1.084093, 1.079007, 
        1.078323
    ],
    "50-55": [
        1.521765, 1.476809, 1.517955, 1.467311, 1.412068, 1.225817, 
        1.163538, 1.147241, 1.161143, 1.177416, 1.227379, 1.240702, 
        1.341752, 1.264172, 1.260207, 1.348483, 1.245898, 1.224986, 
        1.238293, 1.223924, 1.228064, 1.206360, 1.179837, 1.170511, 
        1.159021
    ],
    "56-99": [
        1.424273, 1.426247, 1.452837, 1.420182, 1.369796, 1.255683, 
        1.218954, 1.192716, 1.181098, 1.185274, 1.205601, 1.208697, 
        1.201661, 1.181862, 1.208291, 1.161377, 1.161642, 1.189040, 
        1.209286, 1.218438, 1.213774, 1.214969, 1.211500, 1.212099, 
        1.215759
    ]
}

data_Losses_RgrpGender_FedCustom_LossGroup_Gender_7_lambda02 = {
    "Round": list(range(0, 25)),
    "M": [
        1.450404, 1.436996, 1.471014, 1.428020, 1.370040, 1.193151, 
        1.110921, 1.088883, 1.089760, 1.107925, 1.154684, 1.167000, 
        1.247925, 1.182644, 1.181286, 1.246457, 1.163079, 1.148335, 
        1.155336, 1.153323, 1.156005, 1.132939, 1.109495, 1.105485, 
        1.102771
    ],
    "F": [
        1.464079, 1.449951, 1.482799, 1.442303, 1.393526, 1.249730, 
        1.203381, 1.193564, 1.204341, 1.224118, 1.270734, 1.279348, 
        1.334832, 1.281777, 1.289430, 1.325857, 1.264159, 1.252234, 
        1.267492, 1.261589, 1.266308, 1.242635, 1.222876, 1.216816, 
        1.214818
    ]
}

# --------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12, 9))

# Subplot 1
ax1.plot(data_Losses_RgrpActivity_FedAvg["Round"], data_Losses_RgrpActivity_FedAvg["Ativos"], label="Ativos", linestyle='-')
ax1.plot(data_Losses_RgrpActivity_FedAvg["Round"], data_Losses_RgrpActivity_FedAvg["Inativos"], label="Inativos", linestyle='-')
ax1.set_ylabel(r"FedAvg", fontsize=14)
ax1.set_title(r"Perdas de Grupo (Atividade)")
ax1.legend()

# Subplot 2
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["00-17"], label="00-17", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["18-24"], label="18-24", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["25-34"], label="25-34", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["35-44"], label="35-44", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["45-49"], label="45-49", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["50-55"], label="50-55", linestyle='-')
ax2.plot(data_Losses_RgrpAge_FedAvg["Round"], data_Losses_RgrpAge_FedAvg["56-99"], label="56-99", linestyle='-')
ax2.set_title(r"Perdas de Grupo (Idade)")
ax2.legend()

# Subplot 3
ax3.plot(data_Losses_RgrpGender_FedAvg["Round"], data_Losses_RgrpGender_FedAvg["M"], label="M", linestyle='-')
ax3.plot(data_Losses_RgrpGender_FedAvg["Round"], data_Losses_RgrpGender_FedAvg["F"], label="F", linestyle='-')
ax3.set_title(r"Perdas de Grupo (Gênero)")
ax3.legend()

# Subplot 4
ax4.plot(data_Losses_RgrpActivity_FedCustom_LossIndv_1["Round"], data_Losses_RgrpActivity_FedCustom_LossIndv_1["Ativos"], label="Ativos", linestyle='-')
ax4.plot(data_Losses_RgrpActivity_FedCustom_LossIndv_1["Round"], data_Losses_RgrpActivity_FedCustom_LossIndv_1["Inativos"], label="Inativos", linestyle='-')
ax4.set_xlabel("Round")
ax4.set_ylabel(r"Fed($\ell$)", fontsize=14)
ax4.legend()

# Subplot 5
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["00-17"], label="00-17", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["18-24"], label="18-24", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["25-34"], label="25-34", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["35-44"], label="35-44", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["45-49"], label="45-49", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["50-55"], label="50-55", linestyle='-')
ax5.plot(data_Losses_RgrpAge_FedCustom_LossIndv_1["Round"], data_Losses_RgrpAge_FedCustom_LossIndv_1["56-99"], label="56-99", linestyle='-')
ax5.set_xlabel("Round")
ax5.legend()

# Subplot 6
ax6.plot(data_Losses_RgrpGender_FedCustom_LossIndv_1["Round"], data_Losses_RgrpGender_FedCustom_LossIndv_1["M"], label="M", linestyle='-')
ax6.plot(data_Losses_RgrpGender_FedCustom_LossIndv_1["Round"], data_Losses_RgrpGender_FedCustom_LossIndv_1["F"], label="F", linestyle='-')
ax6.set_xlabel("Round")
ax6.legend()

# Subplot 7
ax7.plot(data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02["Round"], data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02["Ativos"], label="Ativos", linestyle='-')
ax7.plot(data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02["Round"], data_Losses_RgrpActivity_FedCustom_LossGroup_Activity_7_lambda02["Inativos"], label="Inativos", linestyle='-')
ax7.set_xlabel("Round")
ax7.set_ylabel(r"FairFed$(\lambda\ell)$", fontsize=14)
ax7.legend()
ax7.text(0.5, 0.95, r'$\lambda = 0.2$', transform=ax7.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

# Subplot 8
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["00-17"], label="00-17", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["18-24"], label="18-24", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["25-34"], label="25-34", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["35-44"], label="35-44", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["45-49"], label="45-49", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["50-55"], label="50-55", linestyle='-')
ax8.plot(data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["Round"], data_Losses_RgrpAge_FedCustom_LossGroup_Age_7_lambda02["56-99"], label="56-99", linestyle='-')
ax8.set_xlabel("Round")
ax8.legend()
ax8.text(0.5, 0.95, r'$\lambda = 0.2$', transform=ax8.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

# Subplot 9
ax9.plot(data_Losses_RgrpGender_FedCustom_LossGroup_Gender_7_lambda02["Round"], data_Losses_RgrpGender_FedCustom_LossGroup_Gender_7_lambda02["M"], label="M", linestyle='-')
ax9.plot(data_Losses_RgrpGender_FedCustom_LossGroup_Gender_7_lambda02["Round"], data_Losses_RgrpGender_FedCustom_LossGroup_Gender_7_lambda02["F"], label="F", linestyle='-')
ax9.set_xlabel("Round")
ax9.legend()
ax9.text(0.5, 0.95, r'$\lambda = 0.6$', transform=ax9.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
