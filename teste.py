class FederatedLearningSystem:
    def __init__(self, lambda_fairness_values):
        self.lambda_fairness_values = lambda_fairness_values
        self.current_lambda_index = 0
        self.lambda_fairness = lambda_fairness_values[self.current_lambda_index]

    def fairness_regularization(self, server_round, loss, global_mean_loss, global_groups_variance, target_variance=0.0005):
        """Calcula a penalidade de fairness, normalizando global_groups_variance."""
        
        # Normalizar global_groups_variance para o intervalo [0.2, 0.8]
        min_var = 1e-5
        max_var = 0.001
        
        if global_groups_variance < min_var:
            normalized_variance = 0.2  # Se estiver abaixo do mínimo, atribui o mínimo
        elif global_groups_variance > max_var:
            normalized_variance = 0.8  # Se estiver acima do máximo, atribui o máximo
        else:
            normalized_range = (global_groups_variance - min_var) / (max_var - min_var)
            normalized_variance = 0.2 + normalized_range * (0.8 - 0.2)

        # Calcular a diferença entre a variância atual dos grupos e o valor alvo
        variance_difference = global_groups_variance - target_variance

        # Cálculo da penalidade de fairness
        diff_loss_global_mean = loss - global_mean_loss
        fairness_penalty = diff_loss_global_mean * ((self.lambda_fairness ** 2) + normalized_variance + variance_difference)
        loss_adjusted = max(0, loss + fairness_penalty - (self.lambda_fairness * 0.1))

        return loss_adjusted

    def train(self, rounds, losses, global_mean_losses, global_group_variances, rgrp_values):
        for round in range(rounds):
            # Simular o cálculo da perda ajustada
            loss = losses[round]
            global_mean_loss = global_mean_losses[round]
            global_group_variance = global_group_variances[round]
            
            adjusted_loss = self.fairness_regularization(round, loss, global_mean_loss, global_group_variance)

            # Monitorar e ajustar lambda_fairness com base em Rgrp
            current_rgrp = rgrp_values[round]
            
            if round > 0 and current_rgrp >= rgrp_values[round - 1]:
                if self.current_lambda_index < len(self.lambda_fairness_values) - 1:
                    self.current_lambda_index += 1
                    self.lambda_fairness = self.lambda_fairness_values[self.current_lambda_index]
                    print(f"Increase lambda_fairness to {self.lambda_fairness} at round {round}")
            else:
                if self.current_lambda_index > 0:
                    self.current_lambda_index -= 1
                    self.lambda_fairness = self.lambda_fairness_values[self.current_lambda_index]
                    print(f"Decrease lambda_fairness to {self.lambda_fairness} at round {round}")

            rgrp_values[round] = current_rgrp


data_rgrp_activity_FedCustom_LossGroup_Activity_7g_lambda02_1 = {
    "Round": list(range(0, 25)),
    "RgrpActivity": [
        0.0026173396412436803, 0.0006352703719567256, 0.00019591878909564766,
        1.0604681092012191e-05, 0.00012984927635120513, 0.0001584007944787266,
        0.0001534710373453147, 0.00011010696269509423, 0.00010502894582993705,
        0.00012174965373278057, 0.00011805157377305336, 0.00019139304626189196,
        0.0002163159406142521, 0.00024524903682131163, 0.0003566562320136005,
        0.000430115345367835, 0.0005184752083206797, 0.00035210707232972863,
        0.0005358160866644293, 0.0009474822310493483, 0.0009437604916978199,
        0.0009298626017292116, 0.00104522087910334, 0.0007149688696862708,
        0.0012414210076472985
    ]
}

# Exemplo de uso
lambda_fairness_values = [0.2, 0.4, 0.6, 0.8]
federated_system = FederatedLearningSystem(lambda_fairness_values)

# Dados de entrada simulados para as rodadas
rounds = 25
losses = [0.1] * rounds  # Substitua pelos valores reais de perda
global_mean_losses = [0.1] * rounds  # Substitua pelos valores reais de perda média global
global_group_variances = [0.0001] * rounds  # Substitua pelos valores reais de variância global dos grupos
rgrp_values = data_rgrp_activity_FedCustom_LossGroup_Activity_7g_lambda02_1["RgrpActivity"]

federated_system.train(rounds, losses, global_mean_losses, global_group_variances, rgrp_values)
