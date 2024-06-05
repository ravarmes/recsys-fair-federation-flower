import pandas as pd

# Histórico de perdas para aprendizado distribuído
data_distributed = {
    "Round": list(range(1, 25)),
    "Loss": [
        1.2272162745048556, 1.2618039625581414, 1.2374676358071046, 1.1741040018206514, 
        1.1052727469224606, 1.064799431188062, 1.0411960910891997, 1.0274921354575093, 
        1.0269259030962032, 1.0285457368802333, 1.035851575504875, 1.0456870250647174, 
        1.053773746524804, 1.0621911385365346, 1.0637408972593587, 1.065360599943739, 
        1.0635859301153399, 1.064267940354373, 1.0670108166834416, 1.0689650875681, 
        1.074843548375729, 1.0876522343099124, 1.0919036435301688, 1.1037369306378115
    ]
}
df_distributed = pd.DataFrame(data_distributed)

# Histórico de perdas para aprendizado centralizado
data_centralized = {
    "Round": list(range(0, 25)),
    "Loss": [
        1.3560837845731255, 1.1964769480836313, 1.2319617446585995, 1.2064436121492197, 
        1.138320341459568, 1.0619875606124765, 1.0152901126355525, 0.9876619316884224, 
        0.9747340205982821, 0.9750976559539505, 0.9795825737497664, 0.9893339148519055, 
        1.0002006924902367, 1.0087330466666757, 1.016263168872587, 1.0170140815590392, 
        1.017192038873963, 1.0142086189215547, 1.012471314001557, 1.0137975647157391, 
        1.0136588445759767, 1.0175884435607108, 1.0285364786619382, 1.032140553737713, 
        1.0425255044998711
    ]
}
df_centralized = pd.DataFrame(data_centralized)

# Métricas por grupo de atividade
data_rgrp_activity = {
    "Round": list(range(0, 25)),
    "RgrpActivity": [
        0.000981994, 0.000238225, 0.000504384, 0.000430394, 0.000188408, 3.596e-06, 
        0.000134395, 0.000476567, 0.000709533, 0.000955670, 0.001092158, 0.001086757, 
        0.001186913, 0.001348532, 0.001563960, 0.001856400, 0.002241598, 0.002600655, 
        0.003121024, 0.003715837, 0.004516703, 0.005286370, 0.006090524, 0.006575472, 
        0.007342683
    ]
}
df_rgrp_activity = pd.DataFrame(data_rgrp_activity)

# Métricas por grupo de idade
data_rgrp_age = {
    "Round": list(range(0, 25)),
    "RgrpAge": [
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
        0.01261870780876268,
    ]
}
df_rgrp_age = pd.DataFrame(data_rgrp_age)

# Métricas por grupo de gênero
data_rgrp_gender = {
    "Round": list(range(0, 25)),
    "RgrpGender": [
        9.39706e-06, 0.000111572, 9.23792e-05, 0.000143523, 0.000352234, 0.000953561, 
        0.00190807, 0.00279572, 0.00327143, 0.00362377, 0.00381965, 0.00389468, 
        0.00402243, 0.00412658, 0.00431023, 0.00446346, 0.00467865, 0.00481244, 
        0.00503336, 0.00527175, 0.00548703, 0.00571105, 0.00594851, 0.00605597, 
        0.00620417
    ]
}
df_rgrp_gender = pd.DataFrame(data_rgrp_gender)

# Métricas de precisão e revocação (precision & recall) para top 10
data_accuracy = {
    "Round": list(range(0, 25)),
    "Accuracy": [
        0.290852, 0.480441, 0.292818, 0.356566, 0.470196, 0.472524, 0.476871, 
        0.488202, 0.488306, 0.488927, 0.495860, 0.514902, 0.524475, 0.525975, 
        0.528200, 0.528304, 0.531046, 0.525717, 0.520180, 0.516248, 0.515730, 
        0.513350, 0.510245, 0.509780, 0.509831
    ],
    "Precision_at_10": [
        0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.4, 0.6, 0.8, 0.4, 0.5, 0.8, 0.6, 0.6, 0.6,
        0.7, 0.9, 0.8, 0.9, 0.9, 0.8, 0.7, 0.5, 0.8, 0.7
    ],
    "Recall_at_10": [
        0.0, 0.0, 0.0, 0.0, 0.285714, 0.875, 0.8, 0.857143, 1.0, 0.571429, 0.833333, 
        1.0, 1.0, 0.857143, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
}
df_accuracy_metrics = pd.DataFrame(data_accuracy)

# Dados fornecidos
data_rmse = {
    "Round": list(range(0, 25)),
    "rmse": [
        1.1645201444625854, 1.0938423871994028, 1.0841187238693237, 1.0621104249954224, 
        1.0424585342407227, 1.0179275274276733, 1.0081915855407715, 0.9968542456626892, 
        0.9968542456626892, 0.9968542456626892, 0.9968542456626892, 1.0170716047286987, 
        1.0311692953109741, 1.0467497110366821, 1.0623263120651245, 1.0702736377716064, 
        1.0768852233886719, 1.0817430019378662, 1.0835092067718506, 1.0842232704162598, 
        1.0842232704162598, 1.0842232704162598, 1.0842232704162598, 1.0842232704162598, 
        1.0842232704162598
    ]
}
df_rmse = pd.DataFrame(data_rmse)

# Processar logs fornecidos
logs = """
Round 1: loss=1.2272162745048556, accuracy=0.290852, precision@10=0.0, recall@10=0.0, rmse=1.1645201444625854, rgrp_activity=0.000981994, rgrp_age=0.003396680190163166, rgrp_gender=9.39706e-06
Round 2: loss=1.2618039625581414, accuracy=0.480441, precision@10=0.0, recall@10=0.0, rmse=1.0938423871994028, rgrp_activity=0.000238225, rgrp_age=0.003883012178654958, rgrp_gender=0.000111572
Round 3: loss=1.2374676358071046, accuracy=0.292818, precision@10=0.0, recall@10=0.0, rmse=1.0841187238693237, rgrp_activity=0.000504384, rgrp_age=0.0032389046429972, rgrp_gender=9.23792e-05
Round 4: loss=1.1741040018206514, accuracy=0.356566, precision@10=0.0, recall@10=0.0, rmse=1.0621104249954224, rgrp_activity=0.000430394, rgrp_age=0.0033415192432913977, rgrp_gender=0.000143523
Round 5: loss=1.1052727469224606, accuracy=0.470196, precision@10=0.2, recall@10=0.285714, rmse=1.0424585342407227, rgrp_activity=0.000188408, rgrp_age=0.003622837286258953, rgrp_gender=0.000352234
Round 6: loss=1.064799431188062, accuracy=0.472524, precision@10=0.7, recall@10=0.875, rmse=1.0179275274276733, rgrp_activity=3.596e-06, rgrp_age=0.004139421103580343, rgrp_gender=0.000953561
Round 7: loss=1.0411960910891997, accuracy=0.476871, precision@10=0.4, recall@10=0.8, rmse=1.0081915855407715, rgrp_activity=0.000134395, rgrp_age=0.004522999975863927, rgrp_gender=0.00190807
Round 8: loss=1.0274921354575093, accuracy=0.488202, precision@10=0.6, recall@10=0.857143, rmse=0.9968542456626892, rgrp_activity=0.000476567, rgrp_age=0.004788569180378254, rgrp_gender=0.00279572
Round 9: loss=1.0269259030962032, accuracy=0.488306, precision@10=0.8, recall@10=1.0, rmse=0.9968542456626892, rgrp_activity=0.000709533, rgrp_age=0.004456227536866096, rgrp_gender=0.00327143
Round 10: loss=1.0285457368802333, accuracy=0.488927, precision@10=0.4, recall@10=0.571429, rmse=0.9968542456626892, rgrp_activity=0.000955670, rgrp_age=0.004230801833006068, rgrp_gender=0.00362377
Round 11: loss=1.035851575504875, accuracy=0.495860, precision@10=0.5, recall@10=0.833333, rmse=0.9968542456626892, rgrp_activity=0.001092158, rgrp_age=0.0039457243582087425, rgrp_gender=0.00381965
Round 12: loss=1.0456870250647174, accuracy=0.514902, precision@10=0.8, recall@10=1.0, rmse=1.0170716047286987, rgrp_activity=0.001086757, rgrp_age=0.003498848430346345, rgrp_gender=0.00389468
Round 13: loss=1.053773746524804, accuracy=0.524475, precision@10=0.6, recall@10=1.0, rmse=1.0311692953109741, rgrp_activity=0.001186913, rgrp_age=0.003322188539265435, rgrp_gender=0.00402243
Round 14: loss=1.0621911385365346, accuracy=0.525975, precision@10=0.6, recall@10=0.857143, rmse=1.0467497110366821, rgrp_activity=0.001348532, rgrp_age=0.0033420332522783544, rgrp_gender=0.00412658
Round 15: loss=1.0637408972593587, accuracy=0.528200, precision@10=0.6, recall@10=1.0, rmse=1.0623263120651245, rgrp_activity=0.001563960, rgrp_age=0.0036406929257304195, rgrp_gender=0.00431023
Round 16: loss=1.065360599943739, accuracy=0.528304, precision@10=0.7, recall@10=1.0, rmse=1.0702736377716064, rgrp_activity=0.001856400, rgrp_age=0.003894552862944088, rgrp_gender=0.00446346
Round 17: loss=1.0635859301153399, accuracy=0.531046, precision@10=0.9, recall@10=1.0, rmse=1.0768852233886719, rgrp_activity=0.002241598, rgrp_age=0.004579852897645133, rgrp_gender=0.00467865
Round 18: loss=1.064267940354373, accuracy=0.525717, precision@10=0.8, recall@10=1.0, rmse=1.0817430019378662, rgrp_activity=0.002600655, rgrp_age=0.005006362570400268, rgrp_gender=0.00481244
Round 19: loss=1.0670108166834416, accuracy=0.520180, precision@10=0.9, recall@10=1.0, rmse=1.0835092067718506, rgrp_activity=0.003121024, rgrp_age=0.005874461016865226, rgrp_gender=0.00503336
Round 20: loss=1.0689650875681, accuracy=0.516248, precision@10=0.9, recall@10=1.0, rmse=1.0842232704162598, rgrp_activity=0.003715837, rgrp_age=0.006780175495069189, rgrp_gender=0.00527175
Round 21: loss=1.074843548375729, accuracy=0.515730, precision@10=0.8, recall@10=1.0, rmse=1.0842232704162598, rgrp_activity=0.004516703, rgrp_age=0.00815280980914856, rgrp_gender=0.00548703
Round 22: loss=1.0876522343099124, accuracy=0.513350, precision@10=0.7, recall@10=1.0, rmse=1.0842232704162598, rgrp_activity=0.005243858, rgrp_age=0.00926234582234116, rgrp_gender=0.00569066
Round 23: loss=1.1040073873828204, accuracy=0.507319, precision@10=0.7, recall@10=1.0, rmse=1.0842232704162598, rgrp_activity=0.006087615, rgrp_age=0.0106762439918514, rgrp_gender=0.00585975
Round 24: loss=1.121217419249214, accuracy=0.504743, precision@10=0.6, recall@10=0.857143, rmse=1.0842232704162598, rgrp_activity=0.006899717, rgrp_age=0.011890893846688748, rgrp_gender=0.0059808
Round 25: loss=1.1354916010599323, accuracy=0.498641, precision@10=0.5, recall@10=1.0, rmse=1.0842232704162598, rgrp_activity=0.007559645, rgrp_age=0.012697250679812908, rgrp_gender=0.00603188
"""

# Criar DataFrame com dados dos logs
data_logs = {
    'round': [],
    'loss': [],
    'accuracy': [],
    'precision@10': [],
    'recall@10': [],
    'rmse': [],
    'rgrp_activity': [],
    'rgrp_age': [],
    'rgrp_gender': []
}

for line in logs.strip().split("\n"):
    parts = line.split(", ")
    data_logs['round'].append(int(parts[0].split(" ")[1][:-1]))
    data_logs['loss'].append(float(parts[0].split("=")[1]))
    data_logs['accuracy'].append(float(parts[1].split("=")[1]))
    data_logs['precision@10'].append(float(parts[2].split("=")[1]))
    data_logs['recall@10'].append(float(parts[3].split("=")[1]))
    data_logs['rmse'].append(float(parts[4].split("=")[1]))
    data_logs['rgrp_activity'].append(float(parts[5].split("=")[1]))
    data_logs['rgrp_age'].append(float(parts[6].split("=")[1]))
    data_logs['rgrp_gender'].append(float(parts[7].split("=")[1]))

df_logs = pd.DataFrame(data_logs)

df_logs.head()
