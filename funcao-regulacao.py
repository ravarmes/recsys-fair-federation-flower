# Grupo 1: Usuário 1 e Usuário 2
# Grupo 2: Usuário 3 e Usuário 4
# Vamos atribuir as seguintes perdas médias:

# Perda média do Grupo 1: 0.2
# Perda média do Grupo 2: 0.3
# Vamos também supor que a global_groups_variance seja 0.1.

lambda_fairness = 0.4
global_groups_variance = 0.1

# Usuário 1 (pertence ao Grupo 1)
local_loss_1 = 0.15  # Suponha que a perda local seja 0.15
group_loss_1 = 0.2   # Perda média do Grupo 1
fairness_penalty_1 = (lambda_fairness * group_loss_1) / global_groups_variance
print("Penalidade de Equidade para Usuário 1:", fairness_penalty_1)

# Usuário 2 (pertence ao Grupo 1)
local_loss_2 = 0.25  # Suponha que a perda local seja 0.25
group_loss_2 = 0.2   # Perda média do Grupo 1
fairness_penalty_2 = (lambda_fairness * group_loss_2) / global_groups_variance
print("Penalidade de Equidade para Usuário 2:", fairness_penalty_2)

# Usuário 3 (pertence ao Grupo 2)
local_loss_3 = 0.35  # Suponha que a perda local seja 0.35
group_loss_3 = 0.3   # Perda média do Grupo 2
fairness_penalty_3 = (lambda_fairness * group_loss_3) / global_groups_variance
print("Penalidade de Equidade para Usuário 3:", fairness_penalty_3)

# Usuário 4 (pertence ao Grupo 2)
local_loss_4 = 0.45  # Suponha que a perda local seja 0.45
group_loss_4 = 0.3   # Perda média do Grupo 2
fairness_penalty_4 = (lambda_fairness * group_loss_4) / global_groups_variance
print("Penalidade de Equidade para Usuário 4:", fairness_penalty_4)



