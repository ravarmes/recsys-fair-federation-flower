from collections import OrderedDict
import random
from typing import Dict, List, Optional, Tuple, Union

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

# Fixing random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Force PyTorch to use deterministic algorithms
torch.use_deterministic_algorithms(True)

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

NUM_CLIENTS = 300

def verificar_trainloaders(trainloaders):
    """Verifica e imprime o conteúdo de cada DataLoader."""
    for i, trainloader in enumerate(trainloaders):
        if i == 0:
            print(f"Trainloader {i+1} (Cliente {i+1}):")
            for data in trainloader:
                inputs, labels = data
                print("Inputs (Usuário, Item):", inputs)
                print("Labels (Avaliações):", labels)
                print() 
            print("============== Fim do DataLoader ============")
            print()

def set_random_seed(seed: int):
    """Função para configurar as sementes para reprodutibilidade."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para GPUs com múltiplas GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets(num_clients: int, filename: str, seed: int = 42):
    """Carrega e divide datasets para os clientes."""
    set_random_seed(seed)

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
        ds_train, ds_val = random_split(dataset, [len_train, len_val], generator=torch.Generator().manual_seed(seed))

        batch_size = 32 if cliente_id <= 14 else 16  # Definindo o tamanho do lote de acordo com o nível de atividade dos usuários
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        
        # Adicionar dados de cada cliente para seleção de teste
        testloader_data.extend(dados_cliente)

    # Supondo que queiramos usar apenas 10% dos dados acumulados para o teste
    num_test_samples = len(testloader_data)
    test_data_sample = random.sample(testloader_data, num_test_samples // 8)  # 10% dos dados para teste

    # Separar dados de teste para evitar sobreposição
    test_data_set = set(map(tuple, test_data_sample))

    # Modificação para acessar os tensores do dataset original
    train_val_data_set = set(
        map(tuple, sum([loader.dataset.dataset.tensors[0].numpy().tolist() for loader in trainloaders + valloaders], []))
    )

    # Garantir que os dados de teste não estejam nos dados de treinamento e validação
    final_test_data = [dados for dados in testloader_data if tuple(dados) not in train_val_data_set]
    
    if len(final_test_data) > 0:
        final_test_sample = random.sample(final_test_data, min(len(final_test_data), num_test_samples // 8))
    else:
        final_test_sample = []

    X_test_all = np.array(final_test_sample)[:, :2]
    y_test_all = np.array(final_test_sample)[:, 2]
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
        user_idx = x[:, 0].long()  # Converter para índice inteiro
        item_idx = x[:, 1].long()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        x = torch.cat((user_embed, item_embed), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * 4 + 1  # Mapeia saída para o intervalo de 1 a 5
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
    parameters_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    net.load_state_dict(parameters_dict, strict=True)

def train(net, trainloader, cid, epochs: int, lotes_por_rodada: int, learning_rate: float):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        num_examples = 0
        for i, (data, targets) in enumerate(trainloader):
            if i >= lotes_por_rodada:
                break
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
            num_examples += data.size(0)
            num_batches += 1
        epoch_loss /= num_examples
        print(f"Client {cid}, Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

def test(net, testloader):
    criterion = nn.MSELoss()
    net.eval()
    test_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = net(data)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item() * data.size(0)
            total += data.size(0)
            correct += (outputs.round() == targets).sum().item()
    test_loss /= total
    accuracy = correct / total
    return test_loss, accuracy

def calculate_f1_recall_at_k(predictions, targets, k):
    relevant_items = targets.argsort()[:, -k:]  # Últimos k itens são os relevantes
    top_k_items = predictions.argsort()[:, -k:]  # Top k itens preditos pelo modelo

    tp = np.array([len(set(relevant_items[i]) & set(top_k_items[i])) for i in range(len(targets))])
    precision = tp / k
    recall = tp / k
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    avg_f1_score = np.nanmean(f1_score)
    avg_recall = np.mean(recall)
    return avg_f1_score, avg_recall

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.group = random.choice(["A", "B", "C"])  # Exemplo: cada cliente pertence a um grupo

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self.net)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.net, parameters)
        epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        lotes_por_rodada = config["lotes_por_rodada"]
        train(self.net, self.trainloader, self.cid, epochs, lotes_por_rodada, learning_rate)
        local_loss, accuracy = test(self.net, self.valloader)
        metrics = {"loss": local_loss, "accuracy": accuracy, "group": self.group, "li": self.cid}  # Adicionando "group" e "li" às métricas
        return get_parameters(self.net), len(self.trainloader), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> fl.client.Client:
    net = Net(300, 1000).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

class GroupFairnessStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit=0.1, fraction_eval=0.1, min_fit_clients=2, min_eval_clients=2,
                 min_available_clients=2, evaluate_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None):
        super().__init__(fraction_fit=fraction_fit, fraction_eval=fraction_eval,
                         min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients,
                         min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn)
        self.rgrp_weights = None
        self.parameters_last_round = None
        self.local_losses = []

    def aggregate_fit(self, rnd, results, failures):
        print(f"Aggregator: aggregate_fit round {rnd}")
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        self.local_losses = [metrics["loss"] for _, _, metrics in results]
        print(f"Local losses: {self.local_losses}")
        self.rgrp_weights = self.calculate_group_weights()
        print(f"Group weights: {self.rgrp_weights}")
        return aggregated_parameters

    def calculate_group_weights(self):
        if len(self.local_losses) == 0:
            return None
        max_loss = max(self.local_losses)
        min_loss = min(self.local_losses)
        range_loss = max_loss - min_loss
        weights = [(max_loss - loss) / range_loss if range_loss > 0 else 1.0 for loss in self.local_losses]
        return weights

    def aggregate_parameters(self, rnd, parameters, client_params, server_params):
        if self.rgrp_weights is None:
            return super().aggregate_parameters(rnd, parameters, client_params, server_params)
        scaled_params = [
            [p * w for p, w in zip(client_param, self.rgrp_weights)]
            for client_param in client_params
        ]
        return super().aggregate_parameters(rnd, parameters, scaled_params, server_params)

def main():
    strategy = GroupFairnessStrategy(
        fraction_fit=0.1,
        fraction_eval=0.1,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=None,
        on_fit_config_fn=configure_fit,
        on_evaluate_config_fn=configure_evaluate,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 2},
        strategy=strategy,
    )

def configure_fit(rnd: int):
    config = {
        "server_round": rnd,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "lotes_por_rodada": 5,
    }
    return config

def configure_evaluate(rnd: int):
    config = {
        "val_steps": 5,
    }
    return config

if __name__ == "__main__":
    main()
