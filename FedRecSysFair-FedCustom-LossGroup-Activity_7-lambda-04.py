# Neste script a estratégia de agregação é ponderar os pesos na proporção das Perdas Individuais
# Além disso, os grupos recebem parâmetros de taxa de aprendizagem e épocas de acordo com as Perdas Individuais dentro do Grupo
# Grupos favorecidos tem menores taxas de aprendizado e épocas
# Grupos desfavorecidos tem maiores taxas de aprendizado e épocas
# As configurações de taxas de aprendizado e épocas são DINÂMICAS (3 configurações)
# A configuração de grupo considerada é a Atividade (Activity)

# Esta classe FedCustom agora inclui as estratégias sugeridas de taxa de aprendizado adaptativa e regularização de equidade, e simplifica a configuração de treinamento para aplicar as melhorias sugeridas de forma coerente.

# !pip install -q flwr[simulation] torch torchvision

from collections import OrderedDict
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from collections import OrderedDict
from AlgorithmUserFairness import GroupLossVariance


import flwr as fl

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 300

datasets = None

# Função para imprimir o conteúdo de cada DataLoader
def verificar_trainloaders(trainloaders):
    for i, trainloader in enumerate(trainloaders):
        #if i == 0 or i == 299:
        if i == 0:
            print(f"Trainloader {i+1} (Cliente {i+1}):")
            for data in trainloader:
                inputs, labels = data
                print("Inputs (Usuário, Item):", inputs)
                print("Labels (Avaliações):", labels)
                print()  # Adiciona uma linha em branco para melhor visualização
            print("============== Fim do DataLoader ============")
            print()  # Linha extra para separar cada DataLoader

            # Comente a próxima linha para visualizar todos os DataLoader
            # break  # Remova ou comente esta linha para verificar todos os trainloaders


def load_datasets(num_clients: int, filename: str):
    # Carregar dados do arquivo Excel
    df = pd.read_excel(filename, index_col=0)
    dados = df.fillna(0).values
    X, y = np.nonzero(dados)
    ratings = dados[X, y]
    # Criar dicionário para agrupar avaliações por usuário
    cliente_avaliacoes = {usuario: [] for usuario in np.unique(X)}
    for usuario, item, rating in zip(X, y, ratings):
        cliente_avaliacoes[usuario].append((usuario, item, rating))
    trainloaders = []
    valloaders = []
    testloader_data = []
    for cliente_id in sorted(cliente_avaliacoes.keys()):
        dados_cliente = np.array(cliente_avaliacoes[cliente_id])
        X_train = dados_cliente[:, :2]
        y_train = dados_cliente[:, 2]
        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        len_val = len(dataset) // 10
        len_train = len(dataset) - len_val
        ds_train, ds_val = random_split(dataset, [len_train, len_val], torch.Generator().manual_seed(42))

        batch_size = 32 if cliente_id <= 14 else 16 # Configurando o tamanho do lote de acordo com o nível de atividade dos usuários
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        testloader_data.extend(dados_cliente) # Adicionar dados de teste do cliente à lista de testes

    # Supondo que queiramos usar apenas 10% dos dados acumulados para o teste
    num_test_samples = len(testloader_data)
    test_data_sample = random.sample(testloader_data, num_test_samples // 8) # 10: deixei 8 para que o último lote tenha 30 exemplos para calcular recall e f1

    X_test_all = np.array(test_data_sample)[:, :2]
    y_test_all = np.array(test_data_sample)[:, 2]
    test_dataset = TensorDataset(torch.from_numpy(X_test_all).float(), torch.from_numpy(y_test_all).float())
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return df, trainloaders, valloaders, testloader


avaliacoes_df, trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, filename="X.xlsx")
# verificar_trainloaders(trainloaders)


class Net(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Saída final para um valor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_idx = x[:, 0].long()  # Converter para índice inteiro
        item_idx = x[:, 1].long()
        # Obter embeddings e concatenar
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        x = torch.cat((user_embed, item_embed), dim=1)
        # Aplicar camadas com ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Aplicar sigmoid para restringir saída entre 0 e 1, depois converter para 1-5
        x = torch.sigmoid(self.fc3(x)) * 4 + 1  # 0-1 para 1-5
        return x
    
    def predict_all(self, num_users, num_items):
        all_users = torch.arange(num_users).view(-1, 1)
        all_items = torch.arange(num_items).view(1, -1)
        user_idx = all_users.expand(num_users, num_items).contiguous().view(-1)
        item_idx = all_items.expand(num_users, num_items).contiguous().view(-1)
        with torch.no_grad():
            predictions = self.forward(torch.stack([user_idx, item_idx], dim=1).long())
            predictions = predictions.view(num_users, num_items).cpu().numpy()
        df = pd.DataFrame(predictions, index=np.arange(num_users), columns=np.arange(num_items))
        return df


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})  # Certifique-se de usar float32
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, cid, epochs: int, lotes_por_rodada: int, learning_rate : float):
    criterion = nn.MSELoss()  # Função de perda para prever avaliações
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        num_examples = 0
        for i, (data, target) in enumerate(trainloader):
            if i >= lotes_por_rodada:
                break  # Parar após atingir o número de lotes desejado
            num_batches += 1
            num_examples += len(data)
            optimizer.zero_grad()  # Zerando gradientes
            outputs = net(data)    # Previsão do modelo
            target = torch.unsqueeze(target, 1)
            loss = criterion(outputs, target) # Calcular a perda
            loss.backward()   # Passo para trás
            optimizer.step()  # Atualizar parâmetros do modelo
            epoch_loss += loss.item()  # Acumular perda
        print(f"[Cliente {cid}] Número de lotes processados: {num_batches}")
        print(f"[Cliente {cid}] Número de exemplos processados: {num_examples}")
        print(f"[Cliente {cid}] Época {epoch + 1}: loss {epoch_loss}")
    return num_examples, epoch_loss


def test(net, testloader, tolerance=0.7, server=False):
    """Avaliar a rede no conjunto de teste."""
    criterion = nn.MSELoss()  # Critério de perda
    net.eval()  # Modo de avaliação
    total = correct = loss = squared_error = 0.0
    precision_at_10 = recall_at_10 = 0.0
    RgrpActivity = RgrpGender = RgrpAge = 0.0
    RgrpActivity_Losses = RgrpGender_Losses = RgrpAge_Losses = []
    outputs = None
    target = None
    with torch.no_grad():
        for data, target in testloader:
            target = torch.unsqueeze(target, 1)
            outputs = net(data)  # Previsão do modelo
            loss += criterion(outputs, target).item() # Calcular a perda
            squared_error += F.mse_loss(outputs, target, reduction='sum').item() # Calcular erro quadrático
            within_tolerance = (torch.abs(outputs - target) <= tolerance).sum().item() # Calcular precisão considerando a tolerância
            correct += within_tolerance
            total += len(target)  # Total de exemplos processados
    loss /= len(testloader)  # Perda média por exemplo
    rmse = torch.sqrt(torch.tensor(squared_error) / total)  # RMSE - Raiz do Erro Quadrático Médio
    accuracy = correct / total  # Cálculo da precisão considerando a tolerância
    if server == True:
        precision_at_10, recall_at_10 = calculate_f1_recall_at_k(outputs, target, k=10, threshold=3.5)
        RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = calculate_Rgrp(net)
    return loss, rmse.item(), accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses


def calculate_f1_recall_at_k(predictions, targets, k=10, threshold=3.5):
    # Assegurando que os tensores sejam convertidos para arrays NumPy de uma dimensão
    predictions_np = predictions.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    # Ordenar as predições em ordem decrescente e pegar os primeiros k índices
    top_k_indices = np.argsort(predictions_np)[::-1][:k]
    # Usar os índices para derivar os top k targets e predictions
    top_predictions = predictions_np[top_k_indices]
    top_targets = targets_np[top_k_indices]
    # Determinar os índices de elementos relevantes e recomendados
    relevant_indices = np.where(top_targets >= threshold)[0]
    recommended_indices = np.where(top_predictions >= threshold)[0]
    # Intersecção dos dois conjuntos de índices
    relevant_recommended_indices = np.intersect1d(relevant_indices, recommended_indices)
    num_relevant_recommended = len(relevant_recommended_indices)
    num_relevant = len(relevant_indices)
    precision_at_k = num_relevant_recommended / k if k > 0 else 0.0
    recall_at_k = num_relevant_recommended / num_relevant if num_relevant > 0 else 0.0
    return precision_at_k, recall_at_k


def calculate_Rgrp(net):
    recomendacoes_df = net.predict_all(300, 1000)
    omega = ~avaliacoes_df.isnull()
    # Agrupamento por Atividade
    G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, 300))}
    # Agrupamento por Gênero
    G_GENDER = {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 95, 96, 99, 100, 102, 103, 105, 107, 108, 109, 110, 111, 112, 115, 117, 118, 120, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 146, 147, 148, 149, 151, 152, 153, 154, 156, 159, 160, 161, 162, 164, 165, 166, 168, 169, 170, 172, 174, 175, 176, 177, 178, 181, 182, 183, 184, 186, 187, 188, 189, 191, 194, 196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 218, 219, 220, 222, 223, 224, 226, 227, 229, 230, 231, 232, 233, 234, 237, 238, 239, 240, 245, 246, 247, 248, 249, 250, 251, 252, 255, 256, 257, 258, 259, 260, 261, 262, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 291, 292, 293, 294, 295, 296, 297, 298, 299], 2: [14, 25, 31, 34, 35, 42, 63, 73, 81, 91, 92, 97, 98, 101, 104, 106, 113, 114, 116, 119, 121, 122, 133, 144, 145, 150, 155, 157, 158, 163, 167, 171, 173, 179, 180, 185, 190, 192, 193, 195, 197, 199, 213, 214, 215, 216, 217, 221, 225, 228, 235, 236, 241, 242, 243, 244, 253, 254, 264, 290]}
    # Agrupamento por Idade
    G_AGE = {1: [14, 132, 194, 262, 273], 2: [8, 23, 26, 33, 48, 50, 61, 64, 70, 71, 76, 82, 86, 90, 92, 94, 96, 101, 107, 124, 126, 129, 134, 140, 149, 157, 158, 159, 163, 168, 171, 174, 175, 189, 191, 201, 207, 209, 215, 216, 222, 231, 237, 244, 246, 251, 255, 265, 270, 275, 282, 288, 290], 3: [3, 6, 7, 9, 10, 11, 15, 16, 21, 22, 24, 28, 29, 31, 32, 34, 35, 37, 39, 40, 41, 42, 43, 44, 45, 51, 53, 55, 56, 59, 60, 63, 65, 66, 69, 72, 73, 74, 75, 79, 80, 81, 85, 89, 93, 97, 102, 103, 104, 106, 108, 109, 110, 111, 116, 118, 119, 120, 122, 128, 130, 131, 133, 135, 136, 138, 139, 141, 142, 143, 145, 147, 151, 155, 161, 164, 169, 170, 173, 176, 179, 181, 183, 186, 187, 188, 190, 192, 193, 195, 196, 198, 200, 202, 203, 204, 205, 206, 211, 212, 213, 217, 219, 220, 223, 225, 226, 229, 230, 232, 233, 234, 236, 238, 240, 241, 249, 252, 253, 254, 258, 260, 261, 264, 267, 268, 269, 276, 277, 279, 280, 283, 285, 286, 287, 289, 291, 293, 294, 295, 296, 298], 4: [1, 2, 4, 5, 13, 17, 18, 25, 27, 36, 38, 49, 52, 57, 68, 77, 78, 84, 87, 88, 91, 95, 98, 99, 100, 105, 112, 117, 121, 127, 144, 146, 150, 152, 153, 156, 166, 172, 177, 182, 199, 208, 210, 214, 227, 228, 243, 245, 248, 250, 256, 263, 271, 272, 278, 292, 297, 299], 5: [19, 20, 30, 46, 47, 54, 58, 62, 67, 83, 113, 125, 137, 148, 160, 165, 167, 184, 197, 221, 235, 239, 242, 281], 6: [0, 114, 115, 123, 178, 180, 185, 224, 247, 257, 266, 274], 7: [12, 154, 162, 218, 259, 284]}
    glv = GroupLossVariance(avaliacoes_df, omega, G_ACTIVITY, 1) #axis = 1 (0 rows e 1 columns)
    RgrpActivity = glv.evaluate(recomendacoes_df)
    RgrpActivity_Losses = glv.get_losses(recomendacoes_df)
    glv = GroupLossVariance(avaliacoes_df, omega, G_GENDER, 1) #axis = 1 (0 rows e 1 columns)
    RgrpGender = glv.evaluate(recomendacoes_df)
    RgrpGender_Losses = glv.get_losses(recomendacoes_df)
    glv = GroupLossVariance(avaliacoes_df, omega, G_AGE, 1) #axis = 1 (0 rows e 1 columns)
    RgrpAge = glv.evaluate(recomendacoes_df)
    RgrpAge_Losses = glv.get_losses(recomendacoes_df)
    
    print("recomendacoes_df")
    print(recomendacoes_df)

    return RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):

        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        lotes_por_rodada = config["lotes_por_rodada"]

        # Use values provided by the config
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        num_examples, loss = train(self.net, self.trainloader, self.cid, epochs=local_epochs, lotes_por_rodada=lotes_por_rodada, learning_rate=learning_rate)

        # Adicionando dados de cálculos dos clientes
        metrics = {
            "group": 42,  # Id do Grupo
            "li": 3.14,   # Erro Individual
            "loss": loss, # Perda do cliente
        }

        # num_examples = len(self.trainloader.dataset)  # Certifique-se de que esta contagem é correta
        result = get_parameters(self.net), num_examples, metrics
        return result

    def evaluate(self, parameters, config):
        print(f"[Cliente {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(self.net, self.valloader, server=False)
        return float(loss), len(self.valloader), {"rmse": float(rmse), "accuracy": float(accuracy)}


def client_fn(cid) -> FlowerClient:
    net = Net(300, 1000).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    print(f"\n\nTamanho do trainloader do Cliente {cid}: {len(trainloader)}\n\n")

    valloader = valloaders[int(cid)]
    flower_client = FlowerClient(cid, net, trainloader, valloader)
    return flower_client.to_client()


#---------------------------------------------------------------
# DEFININDO UMA ESTRATÉGIA DE AGREGAÇÃO CUSTOMIZADA

from typing import Callable, Union, Optional, List, Dict, Tuple
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    MetricsAggregationFn,
    NDArrays,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class FedCustom(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = NUM_CLIENTS,
        min_evaluate_clients: int = NUM_CLIENTS,
        min_available_clients: int = NUM_CLIENTS,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.loss_avg_per_group = {}
        self.all_losses = []
        self.all_weights = []

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        net = Net(300, 1000)
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def adaptive_learning_rate(self, initial_lr, decay_factor, round_num):
        return initial_lr / (1 + decay_factor * round_num)

    def fairness_regularization(self, loss, global_mean_loss, group_variance, lambda_fairness):
        diff_loss_global_mean = loss - global_mean_loss
        fairness_penalty = lambda_fairness * diff_loss_global_mean * group_variance
        return loss + fairness_penalty

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        config = {
            "server_round": server_round,
            "local_epochs": 20,
            "learning_rate": self.adaptive_learning_rate(0.01, 0.01, server_round),
            "lotes_por_rodada": server_round,
        }

        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, 300))}
        total_loss = sum(fit_res.metrics.get('loss', 0) for _, fit_res in results)

        group_losses = {}
        group_counts = {}
        for group, client_indexes in G_ACTIVITY.items():
            group_loss = sum(fit_res.metrics.get('loss', 0) for index, (client, fit_res) in enumerate(results) if index in client_indexes)
            group_count = sum(1 for index in client_indexes if index < len(results))
            group_losses[group] = group_loss
            group_counts[group] = group_count

        self.loss_avg_per_group = {group: (group_losses[group] / group_counts[group] if group_counts[group] != 0 else 0) for group in G_ACTIVITY}
        print(f"Perda Média por Grupo: {self.loss_avg_per_group}")

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        print(f"Número total de exemplos agregados: {total_examples}")

        group_variance = np.var(list(self.loss_avg_per_group.values()))
        global_mean_loss = np.mean(list(self.loss_avg_per_group.values()))

        fairness_losses = []
        for client_index, (client, fit_res) in enumerate(results):
            local_loss = fit_res.metrics.get('loss', 0)
            fairness_loss = self.fairness_regularization(local_loss, global_mean_loss, group_variance, lambda_fairness=-0.4)
            fairness_losses.append((parameters_to_ndarrays(fit_res.parameters), fairness_loss))

        def aggregate(weights):
            total_weight = sum(1 / weight for _, weight in weights)
            weighted_avg = [np.zeros_like(weight) for weight in weights[0][0]]
            for weight, scale in weights:
                for i in range(len(weighted_avg)):
                    weighted_avg[i] += (weight[i] * (1 / scale) / total_weight)
            return weighted_avg

        parameters_aggregated = ndarrays_to_parameters(aggregate(fairness_losses))
        metrics_aggregated = {}

        for client_index, (client, fit_res) in enumerate(results):
            loss = fit_res.metrics.get('loss', 0)
            weight = loss / total_loss
            print(f"Cliente {client.cid}: Perda = {loss}, Peso = {weight}")
            self.all_losses.append(loss)
            self.all_weights.append(weight)
        
        return parameters_aggregated, metrics_aggregated


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configurar a próxima rodada de avaliação."""
        
        # Se a fração de avaliação for zero, não há avaliação
        if self.fraction_evaluate == 0.0:
            return []
        
        # Cria configuração de avaliação
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Selecionar clientes para avaliação
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Retorna pares cliente/configuração para avaliação
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        
        if not results:  # Se não houver resultados, retorne nada
            return None, {}
        
        # # # Calcular a perda média ponderada
        # # loss_aggregated = weighted_loss_avg(
        # #     [
        # #         (evaluate_res.num_examples, evaluate_res.loss)
        # #         for _, evaluate_res in results
        # #     ]
        # # )

        # # Calcula a perda média ponderada e exibe o número de amostras de cada cliente
        # examples_and_loss = [
        #     (evaluate_res.num_examples, evaluate_res.loss)
        #     for _, evaluate_res in results
        # ]

        # loss_aggregated = weighted_loss_avg(examples_and_loss)

        # # Exibindo o número de amostras consideradas por cada cliente
        # # Certifique-se de ter a lista de clientes e resultados para obter o ID do cliente corretamente
        # for result, (num_samples, _) in zip(results, examples_and_loss):
        #     client_proxy = result[0]  # O primeiro item da tupla deve ser o ClientProxy
        #     client_id = client_proxy.cid  # Obter o ID do cliente real
        #     print(f"Cliente {client_id}: Número de amostras consideradas = {num_samples}")

        
        # Dicionário para métricas agregadas (vazio por enquanto)
        metrics_aggregated = {}
        loss_aggregated = None
        
        # Retorna a perda agregada e métricas
        return loss_aggregated, metrics_aggregated


    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = Net(300, 1000).to(DEVICE)
        set_parameters(net, parameters_to_ndarrays(parameters))
        loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(net, testloader, server=True)
        metrics = {"rmse": rmse, "accuracy": accuracy, "precision_at_10": precision_at_10, "recall_at_10": recall_at_10, "RgrpActivity": RgrpActivity, "RgrpGender": RgrpGender, "RgrpAge": RgrpAge, "RgrpActivity_Losses": RgrpActivity_Losses, "RgrpGender_Losses": RgrpGender_Losses, "RgrpAge_Losses": RgrpAge_Losses}  # Agrupar RMSE e accuracy em um dicionário
        print(f"Server-side evaluation :: Round {server_round}")
        print(f"loss {loss} / RMSE {rmse} / accuracy {accuracy} / Precision@10 {precision_at_10} / Recall@10 {recall_at_10}")
        print(f"RgrpActivity {RgrpActivity} / RgrpGender {RgrpGender} / RgrpAge {RgrpAge}")
        print(f"RgrpActivity_Losses {RgrpActivity_Losses} / RgrpGender_Losses {RgrpGender_Losses} / RgrpAge_Losses {RgrpAge_Losses}")
        return loss, metrics
    
        
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Retorna o tamanho da amostra e o número necessário de clientes para treinamento."""
        
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Usar uma fração dos clientes disponíveis para avaliação."""
        
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


#-----------------------------------------------------------------

# Especifique os recursos do cliente se precisar de GPU (o padrão é 1 CPU e 0 GPU).
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=24),
    strategy=FedCustom(),  # <-- pass the new strategy here
    client_resources=client_resources,
)