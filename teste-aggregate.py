import numpy as np

# Definir resultados fictícios para os clientes
results = [
    (None, FitRes(metrics={'loss': 0.5})),  # Cliente 0
    (None, FitRes(metrics={'loss': 0.8})),  # Cliente 1
    (None, FitRes(metrics={'loss': 0.4})),  # Cliente 2
    (None, FitRes(metrics={'loss': 0.6}))   # Cliente 3
]

# Definir grupos de clientes
G_ACTIVITY = {1: [0, 1], 2: [2, 3]}

# Calcular group_losses e group_counts
group_losses = {}
group_counts = {}
for group, client_indexes in G_ACTIVITY.items():
    group_loss = sum(results[index][1].metrics.get('loss', 0) for index in client_indexes)
    group_count = len(client_indexes)
    group_losses[group] = group_loss
    group_counts[group] = group_count

# Calcular self.loss_avg_per_group
loss_avg_per_group = {group: (group_losses[group] / group_counts[group] if group_counts[group] != 0 else 0) for group in G_ACTIVITY}

# Imprimir resultados
print("group_losses:", group_losses)
print("group_counts:", group_counts)
print("self.loss_avg_per_group:", loss_avg_per_group)

# Calcular fairness_losses
lambda_fairness = 0.4
fairness_losses = []
for client_index, (client, fit_res) in enumerate(results):
    local_loss = fit_res.metrics.get('loss', 0)
    client_group = next(group for group, client_indexes in G_ACTIVITY.items() if client_index in client_indexes)
    
    # Obter a perda média do grupo específico
    group_loss = group_losses[client_group]
    
    # Aplicar a penalidade de justiça aos pesos individuais dos clientes
    fairness_penalty = loss_avg_per_group[client_group]
    fairness_weight = 1.0 + lambda_fairness * fairness_penalty
    
    # Calcular a perda justa do cliente
    fairness_loss = local_loss * fairness_weight
    
    fairness_losses.append((None, fairness_loss))

# Calcular os pesos normalizados
total_fairness_loss = sum(weighted_loss for _, weighted_loss in fairness_losses)
normalized_weights = [(None, weighted_loss / total_fairness_loss) for _, weighted_loss in fairness_losses]

# Imprimir resultados
print("fairness_losses:", fairness_losses)
print("normalized_weights:", normalized_weights)
