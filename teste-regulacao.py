import numpy as np

def fairness_regularization(loss, global_mean_loss, global_groups_variance, lambda_fairness, scaling_factor=None):
    diff_loss_global_mean = loss - global_mean_loss
    
    # Normalização de global_groups_variance (usando z-score)
    if scaling_factor is None:
        mean_diff_loss = np.sqrt(np.mean(diff_loss_global_mean ** 2)) # RMS of diff
        mean_global_groups_variance = np.sqrt(np.mean(global_groups_variance ** 2)) # RMS of variance
        scaling_factor = mean_diff_loss / mean_global_groups_variance if mean_global_groups_variance != 0 else 1
    
    scaled_global_groups_variance = global_groups_variance * scaling_factor
    
    # Calcular a penalidade de justiça
    fairness_penalty = diff_loss_global_mean * (lambda_fairness - scaled_global_groups_variance)
    
    # Garantir que fairness_penalty não introduza valores negativos excessivos
    fairness_penalty = max(fairness_penalty, -diff_loss_global_mean)
    
    adjusted_loss = loss + fairness_penalty
    
    print(f"------ Round ------")
    print(f"loss: {loss}")
    print(f"global_mean_loss: {global_mean_loss}")
    print(f"diff_loss_global_mean: {diff_loss_global_mean}")
    print(f"global_groups_variance: {global_groups_variance}")
    print(f"lambda_fairness: {lambda_fairness}")
    print(f"scaling_factor: {scaling_factor}")
    print(f"scaled_global_groups_variance: {scaled_global_groups_variance}")
    print(f"fairness_penalty: {fairness_penalty}")
    print(f"adjusted_loss: {adjusted_loss}")
    print("------ End of Round ------\n")

    return adjusted_loss

# Dados reais para a análise
loss_list = [1.161932349205017, 0.9239311218261719, 1.4816029071807861, 1.4523099660873413, 0.33788302540779114,
             3.3733596801757812, 1.3808355331420898, 2.1803938150405884]
global_mean_loss = [0.9028764149337485, 0.9028764149337485, 0.9028764149337485, 0.9028764149337485, 0.9028764149337485, 
                    1.791637227527405, 1.791637227527405, 1.791637227527405]
global_groups_variance = [0.0029010745486304475, 0.0029010745486304475, 0.0029010745486304475, 0.0029010745486304475,
                          0.0029010745486304475, 0.0197167352709749, 0.0197167352709749, 0.0197167352709749]
lambda_fairness = 0.2

# Calcular automaticamente o scaling_factor
mean_diff_loss = np.mean([loss - global_mean_loss[i] for i, loss in enumerate(loss_list)])
mean_global_groups_variance = np.mean(global_groups_variance)
scaling_factor = mean_diff_loss / mean_global_groups_variance if mean_global_groups_variance != 0 else 1

# Exemplo de chamada com o scaling_factor calculado
for i, loss in enumerate(loss_list):
    adjusted_loss = fairness_regularization(loss, global_mean_loss[i], global_groups_variance[i], lambda_fairness, scaling_factor)
    print(f'Server_round: {i + 1}, Adjusted_loss: {adjusted_loss}')
