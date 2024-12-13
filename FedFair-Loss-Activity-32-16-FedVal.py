from collections import OrderedDict, defaultdict
import random
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from AlgorithmUserFairness import GroupLossVariance
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
    NDArrays,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativar o uso da GPU

import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignorar mensagens de aviso do TensorFlow

# Fixando seeds para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Forçando o PyTorch a usar algoritmos determinísticos
torch.use_deterministic_algorithms(True)

DEVICE = torch.device("cpu")  # Use "cuda" para treinar na GPU
print(f"Treinando em {DEVICE} usando PyTorch {torch.__version__} e Flower {fl.__version__}")

NUM_CLIENTS = 300

def load_datasets(num_clients: int, filename: str, seed: int = 42):
    """Carrega e divide datasets para os clientes."""
    # Configurando a semente
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Carregando dados do arquivo Excel
    df = pd.read_excel(filename, index_col=0)
    dados = df.fillna(0).values
    X, y = np.nonzero(dados)
    ratings = dados[X, y]
    
    # Criando dicionário para agrupar avaliações por usuário
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
        ds_train, ds_val = random_split(dataset, [len_train, len_val], generator=torch.Generator().manual_seed(seed))

        batch_size = 32 if cliente_id <= 14 else 16  # Tamanho do lote de acordo com a atividade dos usuários
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        
        # Adicionar dados de cada cliente para seleção de teste
        testloader_data.extend(dados_cliente)

    # Selecionando dados de teste
    num_test_samples = len(testloader_data)
    test_data_sample = random.sample(testloader_data, num_test_samples // 8)

    # Separando dados de teste para evitar sobreposição
    test_data_set = set(map(tuple, test_data_sample))

    # Garantir que os dados de teste não estejam nos dados de treinamento e validação
    final_test_data = [dados for dados in testloader_data if tuple(dados) not in set(map(tuple, random.sample(trainloaders, min(len(trainloaders), 20))))]

    if len(final_test_data) > 0:
        final_test_sample = random.sample(final_test_data, min(len(final_test_data), num_test_samples // 8))
    else:
        final_test_sample = []

    X_test_all = np.array(final_test_sample)[:, :2]
    y_test_all = np.array(final_test_sample)[:, 2]
    test_dataset = TensorDataset(torch.from_numpy(X_test_all).float(), torch.from_numpy(y_test_all).float())
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return df, trainloaders, valloaders, testloader

# Carregando os datasets
avaliacoes_df, trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, filename="X.xlsx")

class Net(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Inicializa os parâmetros do modelo usando uma semente fixa."""
        torch.manual_seed(SEED)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_idx = x[:, 0].long()
        item_idx = x[:, 1].long()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        x = torch.cat((user_embed, item_embed), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * 4 + 1
        return x

    def predict_all(self, num_users, num_items):
        all_users = torch.arange(num_users).view(-1, 1)
        all_items = torch.arange(num_items).view(1, -1)
        user_idx = all_users.expand(num_users, num_items).contiguous().view(-1)
        item_idx = all_items.expand(num_users, num_items).contiguous().view(-1)
        with torch.no_grad():
            predictions = self.forward(torch.stack([user_idx, item_idx], dim=1).long())
            predictions = predictions.view(num_users, num_items).cpu().numpy()
        return pd.DataFrame(predictions, index=np.arange(num_users), columns=np.arange(num_items))

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in net.parameters()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.parameters(), parameters)
    for param, tensor in params_dict:
        param.data.copy_(torch.tensor(tensor))

# Implementação do cliente do Flower
class PoisonDetectClient(fl.client.NumPyClient):
    def __init__(self, model: Net, x_test: torch.Tensor, y_test: torch.Tensor, client_id: int):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.client_id = client_id
    
    def get_parameters(self, config) -> List[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        set_parameters(self.model, parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        num_examples, loss = self.train_local()
        metrics = {"loss": loss}
        return self.get_parameters(), num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = self.evaluate_local()
        return float(loss), len(self.y_test), {"accuracy": acc}

    def train_local(self):
        dataset = TensorDataset(self.x_test, self.y_test)
        trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        total_loss = 0
        for data, target in trainloader:
            optimizer.zero_grad()
            output = self.model(data)
            target = target.view(-1, 1)  # Ajustar a dimensão para MSE
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return len(dataset), total_loss / len(trainloader)

    def evaluate_local(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.x_test)
            target = self.y_test.view(-1, 1)
            loss = F.mse_loss(preds, target).item()
            acc = (preds.round() == target).float().mean().item()  # Cálculo de precisão

        return loss, acc

# Classe de estratégia personalizada para agregação
class FedCustom(Strategy):
    """Estratégia personalizada para agregação de modelos."""

    def __init__(self, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 1, min_evaluate_clients: int = 1,
                 min_available_clients: int = 1,
                 s1_overall: float = 2.0, s1_label: float = 3.0, s2: float = 3.0) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.s1_overall = s1_overall
        self.s1_label = s1_label
        self.s2 = s2
    
    def calculate_new_aggregated(self, results: List[Tuple], last_agg_w: List[np.ndarray]):
        label_acc_dict = defaultdict(list)
        nodes_acc = {}
        loss_dict = {}

        for client_id, parameters, loss, acc, label_acc in results:
            label_acc_dict[client_id] = label_acc
            nodes_acc[client_id] = acc
            loss_dict[client_id] = loss

        points = self.get_points_overall(nodes_acc, loss_dict)
        aggregated_weights = self.agg_copy_weights(results, points, last_agg_w)
        return aggregated_weights

    def get_points_overall(self, nodes_acc, loss_dict):
        """Cálculo das pontuações gerais."""
        mean_loss = np.mean(list(loss_dict.values()))
        points = {}
        for client_id, loss in loss_dict.items():
            score = self.s1_overall * (mean_loss - loss)
            points[client_id] = max(0, score)
        return points

    def agg_copy_weights(self, results, points, last_weights):
        """Agrega pesos com base nas pontuações calculadas."""
        aggregated_weights = [np.zeros_like(weights) for weights in last_weights]
        total_points = sum(points.values())

        for (client_id, client_weights) in results:
            weight_factor = points[client_id] / total_points if total_points > 0 else 0
            for i, weight in enumerate(client_weights):
                aggregated_weights[i] += weight_factor * weight 

        # Adicionando pesos do modelo anterior
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] += last_weights[i]

        return aggregated_weights

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {
            "server_round": server_round,
            "local_epochs": 20,
            "learning_rate": 0.01,
        }
        
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if len(results) == 0:
            return None, {}

        last_agg_w = get_parameters(self.model)  # Acessar os pesos do modelo do servidor
        weights_results = [(res.parameters, res.num_examples) for client, res in results]

        aggregated_weights = self.calculate_new_aggregated(weights_results, last_agg_w)
        
        metrics_aggregated = {}
        return aggregated_weights, metrics_aggregated

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


def client_fn(cid) -> PoisonDetectClient:
    """Cria uma instância do cliente."""
    model = Net(num_users=300, num_items=1000).to(DEVICE)  # Criação do modelo
    trainloader = trainloaders[int(cid)]  # Carregando o DataLoader de treinamento para este cliente
    valloader = valloaders[int(cid)]  # Carregando o DataLoader de validação para este cliente
    
    # Ajustando a estrutura para utilizar os dados
    # Montar dados de teste a partir do DataLoader correspondente à validação, se aplicável
    x_val, y_val = [], []
    for data in valloader:
        inputs, labels = data
        x_val.append(inputs)
        y_val.append(labels)

    # Convertendo listas para tensores
    x_val = torch.cat(x_val)
    y_val = torch.cat(y_val)

    # Exibir o tamanho do DataLoader do cliente
    print(f"\n\nTamanho do trainloader do Cliente {cid}: {len(trainloader)}\n\n")
    
    # Criando a instância do cliente do Flower
    flower_client = PoisonDetectClient(model, x_val, y_val, cid)  # Criar cliente com os dados
    return flower_client  # Retorna a instância do cliente

# Lógica para iniciar a simulação com o Flower
if __name__ == "__main__":
    NUM_CLIENTS = 300
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=24),
        strategy=FedCustom(),
    )