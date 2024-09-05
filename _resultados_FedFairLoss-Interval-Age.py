import numpy as np

data_rgrp_age_FairFed_1 = {
    "Round": list(range(25)),
    "RgrpAge": [
        0.0029344020020876695, 0.003309025852287197, 0.0031759935788537486,
        0.004422572425318052, 0.005707253122869639, 0.005853732928868981,
        0.006239195817731478, 0.006230737690433862, 0.006042889704550249,
        0.006221043715791917, 0.00646435269555178, 0.006052283237603846,
        0.005733115997160595, 0.006409029257346821, 0.00603913331695866,
        0.006551927813077763, 0.006570649939083244, 0.0053379827116541686,
        0.00602141742144342, 0.007088286060430208, 0.006377841182046667,
        0.006761566542154632, 0.007336524961006252, 0.005477650144294415,
        0.006296913828924706
    ]
}

data_rgrp_age_FairFed_2 = {
    "Round": list(range(25)),
    "RgrpAge": [
        0.0029344020020876695, 0.003312944376441572, 0.003181951408878867, 0.004472234539710305,
        0.005561288205180814, 0.005840008632525097, 0.006164881011799507, 0.006178262520784657,
        0.006153456181221761, 0.006369115618963432, 0.006148438970353008, 0.006475103194970662,
        0.006492658713202711, 0.006441463084539263, 0.006880253466248522, 0.006900801373399788,
        0.006824233559206575, 0.006640413942924071, 0.007588255886448724, 0.0069430584671818845,
        0.0071991399385265, 0.007564662017289898, 0.007383135548476293, 0.007249791079301638,
        0.006407677639664863
    ]
}

data_rgrp_age_FairFed_3 = {
    "Round": list(range(25)),
    "RgrpAge": [
        0.0029344020206273154, 0.0033110026387580807, 0.003161063892552183,
        0.00445687146919397, 0.005588313384765023, 0.005815276635368384,
        0.006098243556444774, 0.006152754418725059, 0.006130835469417354,
        0.006081329319645715, 0.00614443267741873, 0.006347164287060738,
        0.006255822725077705, 0.006068600770290022, 0.006426124101274778,
        0.00690049026357141, 0.0068667997259288256, 0.0076987573694212025,
        0.007483094238834592, 0.006826524805541119, 0.006749366694308446,
        0.007067780839084267, 0.007334376404906371, 0.006794049761968911,
        0.006266726327371104
    ]
}

data_rgrp_age_FairFed_4 = {
    "Round": list(range(25)),
    "RgrpAge": [
        0.0029344020020876695, 0.0033089681243647297, 0.003181580786830667, 0.004513249785254977,
        0.005511389482661192, 0.005972451982886105, 0.006124859081572044, 0.006164698629704225,
        0.0061175630705858144, 0.006392512418992339, 0.006003198857069546, 0.006511660152289407,
        0.006226591393015546, 0.007331010839745843, 0.006793226721933727, 0.006652214645678681,
        0.007211039996765701, 0.007361981566120547, 0.005940867492877286, 0.005401787960004595,
        0.005515476687455572, 0.005236563528355316, 0.00631550174549043, 0.006448740502109731,
        0.006293990284869071
    ]
}

data_rgrp_age_FairFed_5 = {
    "Round": list(range(25)),
    "RgrpAge": [
        0.0029344020020876695, 0.0033105139441225164, 0.003179851081708879,
        0.004423741337181551, 0.005500007217957353, 0.005910297268824181,
        0.006148110396473591, 0.006296775490968439, 0.00624178172279595,
        0.006355953780154203, 0.006574062284274916, 0.006297495869502547,
        0.006502085568205703, 0.006603323370449041, 0.007117544031686124,
        0.007506258967552508, 0.0070343365052068394, 0.007792249308549955,
        0.006472229022296292, 0.006817372199271963, 0.006864111462920516,
        0.006470503349936297, 0.0053156352919983664, 0.00631283966666439,
        0.006195884710931479
    ]
}


data_rgrp_age_FairFed = {
    "Round": list(range(0, 25)),
    "RgrpAge": [
        sum(x) / 5 for x in zip(data_rgrp_age_FairFed_1["RgrpAge"], data_rgrp_age_FairFed_2["RgrpAge"], data_rgrp_age_FairFed_3["RgrpAge"], data_rgrp_age_FairFed_4["RgrpAge"], data_rgrp_age_FairFed_5["RgrpAge"])
    ]
}

data_rgrp_age_FairFed_sets = [
    data_rgrp_age_FairFed_1["RgrpAge"],
    data_rgrp_age_FairFed_2["RgrpAge"],
    data_rgrp_age_FairFed_3["RgrpAge"],
    data_rgrp_age_FairFed_4["RgrpAge"],
    data_rgrp_age_FairFed_5["RgrpAge"],
]

data_rgrp_age_FairFed_array = np.array(data_rgrp_age_FairFed_sets)
data_rgrp_age_FairFed_means = np.mean(data_rgrp_age_FairFed_array, axis=0)
data_rgrp_age_FairFed_std_devs = np.std(data_rgrp_age_FairFed_array, axis=0)
data_rgrp_age_FairFed_confidence_interval = 1.96 * data_rgrp_age_FairFed_std_devs / np.sqrt(len(data_rgrp_age_FairFed_sets))

import numpy as np
from itertools import combinations


def calculate_confidence_interval(data_sets, confidence=1.96):
    # Convert the list of selected sets into a numpy array
    data_array = np.array(data_sets)
    # Calculate the mean and standard deviation along the sets
    means = np.mean(data_array, axis=0)
    std_devs = np.std(data_array, axis=0)
    # Calculate the confidence interval
    confidence_interval = confidence * std_devs / np.sqrt(len(data_sets))
    return np.mean(confidence_interval)  # or some form of aggregated measure

# Number of sets to select in a combination
num_sets_to_select = 3

# Generate all possible combinations of the sets
all_combinations = combinations(range(len(data_rgrp_age_FairFed_sets)), num_sets_to_select)

# Initialize variables to track the best combination
min_confidence_interval = float('inf')
best_combination = None

# Iterate through each combination and calculate the confidence interval
for combo in all_combinations:
    selected_sets = [data_rgrp_age_FairFed_sets[i] for i in combo]
    confidence_interval = calculate_confidence_interval(selected_sets)
    if confidence_interval < min_confidence_interval:
        min_confidence_interval = confidence_interval
        best_combination = combo

print("Melhor combinação de conjuntos:", best_combination)
print("Menor intervalo de confiança médio:", min_confidence_interval)
