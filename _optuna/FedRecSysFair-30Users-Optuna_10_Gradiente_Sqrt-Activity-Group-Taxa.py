from collections import OrderedDict
import random
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
import json


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

# Fixando sementes para reprodução
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Forçamos o PyTorch a usar algoritmos determinísticos
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cpu")  # Use "cuda" para GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

NUM_CLIENTS = 30
NUM_ROUNDS = 15

RESULTS = []

def verificar_datasets_file(trainloaders, valloaders, testloader):
    with open('datasets.txt', 'w', encoding='utf-8') as file:
        def custom_print(*args, **kwargs):
            print(*args, file=file, **kwargs)
        
        custom_print("============== Trainloaders ============\n")
        for i, trainloader in enumerate(trainloaders):
            if i == 0:
                custom_print(f"Trainloader {i+1} (Cliente {i+1}):")
                for data in trainloader:
                    inputs, labels = data
                    custom_print("Inputs (Usuário, Item):", inputs)
                    custom_print("Labels (Avaliações):", labels)
                    custom_print() 
                custom_print("============== Fim do DataLoader ============")
                custom_print()

        custom_print("\n\n============== Valloader ============\n")
        for i, valloader in enumerate(valloaders):
            if i == 0:
                custom_print(f"Valloader {i+1} (Cliente {i+1}):")
                for data in valloader:
                    inputs, labels = data
                    custom_print("Inputs (Usuário, Item):", inputs)
                    custom_print("Labels (Avaliações):", labels)
                    custom_print() 
                custom_print("============== Fim do DataLoader ============")
                custom_print()           

        custom_print("\n\n============== Testloader ============\n")
        for data in testloader:
            inputs, labels = data
            custom_print("Inputs (Usuário, Item):", inputs)
            custom_print("Labels (Avaliações):", labels)
            custom_print() 
        custom_print("============== Fim do DataLoader ============")
        custom_print()

def set_random_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets(num_clients: int, filename: str, seed: int = 42):
    set_random_seed(seed)
    df = pd.read_excel(filename, index_col=0)
    dados = df.fillna(0).values
    X, y = np.nonzero(dados)
    ratings = dados[X, y]
    
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

        batch_size = 32 if cliente_id <= 14 else 16
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        
        testloader_data.extend(dados_cliente)

    num_test_samples = len(testloader_data)
    test_data_sample = random.sample(testloader_data, num_test_samples // 8)

    test_data_set = set(map(tuple, test_data_sample))

    train_val_data_set = set(
        map(tuple, sum([loader.dataset.dataset.tensors[0].numpy().tolist() for loader in trainloaders + valloaders], []))
    )

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

avaliacoes_df, trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, filename="X-30.xlsx")
# verificar_datasets_file(trainloaders, valloaders, testloader)

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
        # Use a fixed seed for weight initialization
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
        for i, (data, target) in enumerate(trainloader):
            if i >= lotes_por_rodada:
                break
            num_batches += 1
            num_examples += len(data)
            optimizer.zero_grad()
            outputs = net(data)
            target = torch.unsqueeze(target, 1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Cliente {cid}] Número de lotes processados: {num_batches}")
        print(f"[Cliente {cid}] Número de exemplos processados: {num_examples}")
        print(f"[Cliente {cid}] Época {epoch + 1}: loss {epoch_loss}")
    return num_examples, epoch_loss

def test(net, testloader, tolerance=0.7, server=False):
    criterion = nn.MSELoss()
    net.eval()
    total = correct = loss = squared_error = 0.0
    precision_at_10 = recall_at_10 = 0.0
    RgrpActivity = RgrpGender = RgrpAge = 0.0
    RgrpActivity_Losses = RgrpGender_Losses = RgrpAge_Losses = []
    outputs = None
    target = None
    with torch.no_grad():
        for data, target in testloader:
            target = torch.unsqueeze(target, 1)
            outputs = net(data)
            loss += criterion(outputs, target).item()
            squared_error += F.mse_loss(outputs, target, reduction='sum').item()
            within_tolerance = (torch.abs(outputs - target) <= tolerance).sum().item()
            correct += within_tolerance
            total += len(target)
    loss /= len(testloader)
    rmse = torch.sqrt(torch.tensor(squared_error) / total)
    accuracy = correct / total
    if server:
        precision_at_10, recall_at_10 = calculate_f1_recall_at_k(outputs, target, k=10, threshold=3.5)
        RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = calculate_Rgrp(net)
    return loss, rmse.item(), accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses

def calculate_f1_recall_at_k(predictions, targets, k=10, threshold=3.5):
    predictions_np = predictions.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    top_k_indices = np.argsort(predictions_np)[::-1][:k]
    top_predictions = predictions_np[top_k_indices]
    top_targets = targets_np[top_k_indices]
    relevant_indices = np.where(top_targets >= threshold)[0]
    recommended_indices = np.where(top_predictions >= threshold)[0]
    relevant_recommended_indices = np.intersect1d(relevant_indices, recommended_indices)
    num_relevant_recommended = len(relevant_recommended_indices)
    num_relevant = len(relevant_indices)
    precision_at_k = num_relevant_recommended / k if k > 0 else 0.0
    recall_at_k = num_relevant_recommended / num_relevant if num_relevant > 0 else 0.0
    return precision_at_k, recall_at_k

def calculate_Rgrp(net):
    recomendacoes_df = net.predict_all(NUM_CLIENTS, 1000)
    omega = ~avaliacoes_df.isnull()
    G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, NUM_CLIENTS))}
    
    G_GENDER = {
        1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29],
        2: [14, 25]
    }

    G_AGE = {
        1: [28, 29, 14],
        2: [8, 23, 26],
        3: [3, 6, 7, 9, 10, 11, 15, 16, 21],
        4: [1, 2, 4, 5, 13, 17, 18],
        5: [19, 20],
        6: [0, 25, 27],
        7: [12, 22, 24]
    }

    glv = GroupLossVariance(avaliacoes_df, omega, G_ACTIVITY, 1)
    RgrpActivity = glv.evaluate(recomendacoes_df)
    RgrpActivity_Losses = glv.get_losses(recomendacoes_df)

    glv = GroupLossVariance(avaliacoes_df, omega, G_GENDER, 1)
    RgrpGender = glv.evaluate(recomendacoes_df)
    RgrpGender_Losses = glv.get_losses(recomendacoes_df)

    glv = GroupLossVariance(avaliacoes_df, omega, G_AGE, 1)
    RgrpAge = glv.evaluate(recomendacoes_df)
    RgrpAge_Losses = glv.get_losses(recomendacoes_df)
    
    return RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        lotes_por_rodada = config["lotes_por_rodada"]

        set_parameters(self.net, parameters)
        num_examples, loss = train(self.net, self.trainloader, self.cid, 
                                   epochs=local_epochs, 
                                   lotes_por_rodada=lotes_por_rodada, 
                                   learning_rate=learning_rate)
        metrics = {"group": 42, "li": 3.14, "loss": loss}
        result = get_parameters(self.net), num_examples, metrics
        return result

    def evaluate(self, parameters, config):
        print(f"[Cliente {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(self.net, self.valloader, server=False)
        return (float(loss), len(self.valloader), 
                {"rmse": float(rmse), 
                 "accuracy": float(accuracy)})

def client_fn(cid) -> FlowerClient:
    net = Net(NUM_CLIENTS, 1000).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    flower_client = FlowerClient(cid, net, trainloader, valloader)
    return flower_client.to_client()  # Converte a instância de NumPyClient para Client

class FedCustom(fl.server.strategy.Strategy):
    def __init__(self, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0, 
                 min_fit_clients: int = NUM_CLIENTS, min_evaluate_clients: int = NUM_CLIENTS, 
                 min_available_clients: int = NUM_CLIENTS, initial_learning_rate=0.2) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.loss_avg_per_group = {}
        self.all_losses = []
        self.all_weights = []
        self.global_groups_variance = 1
        self.previous_global_loss = 0
        
        # Armazenar os hiperparâmetros como atributos da classe
        self.learning_rate = initial_learning_rate
        self.previous_rgrp_activity = None

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        set_random_seed(SEED)  # Mantém a semente ao inicializar
        net = Net(NUM_CLIENTS, 1000)
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)
  
    def adaptive_learning_rate(self, initial_lr, decay_factor, round_num):
        """Calcula a taxa de aprendizado adaptativa."""
        return initial_lr / (1 + decay_factor * round_num)


    # Função de Regulação com Normalização das Perdas
    def fairness_regularization(self, server_round, client_index, loss, group_mean_loss, global_groups_variance):
        if loss < group_mean_loss:
            adjustment_factor = 0.9
        else:
            adjustment_factor = 1.1
        
        lr = self.adaptive_learning_rate(0.2, 0.01, server_round)
        
        fairness_penalty = group_mean_loss * (global_groups_variance ** 0.25) * adjustment_factor * lr
        adjusted_loss = loss + fairness_penalty

        with open("FedRecSysFair-30Users-Optuna_09_Gradiente_Sqrt-Activity-Group-Taxa.log", "a") as log_file:
            log_file.write("\n\nfairness_regularization -------------------------------\n")
            log_file.write(f"server_round: {server_round}\n")
            log_file.write(f"client_index: {client_index}\n")
            log_file.write(f"loss: {loss}\n")
            log_file.write(f"group_mean_loss: {group_mean_loss}\n")
            log_file.write(f"global_groups_variance: {global_groups_variance}\n")
            log_file.write(f"adjustment_factor: {adjustment_factor}\n")
            log_file.write(f"learning_rate: {self.learning_rate}\n")
            log_file.write(f"lr: {lr}\n")
            log_file.write(f"loss + fairness_penalty: {loss + fairness_penalty}\n")

        return adjusted_loss
    

    # def evaluate_and_adjust_learning_rate(self, new_rgrp_activity):
    #     if self.previous_rgrp_activity is not None:
    #         if new_rgrp_activity > self.previous_rgrp_activity:
    #             # Aumentar a taxa de aprendizado se a injustiça aumenta
    #             self.learning_rate *= 1.05
    #         elif new_rgrp_activity < self.previous_rgrp_activity:
    #             # Diminuir a taxa de aprendizado se a injustiça diminui
    #             self.learning_rate *= 0.95

    #     self.previous_rgrp_activity = new_rgrp_activity



    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        config = {
            "server_round": server_round,
            "local_epochs": 20,
            "learning_rate": self.adaptive_learning_rate(0.01, 0.01, server_round),
            "lotes_por_rodada": server_round,
        }

        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, NUM_CLIENTS))}
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

        global_groups_variance = np.var(list(self.loss_avg_per_group.values()))
        print(f"global_groups_variance: {global_groups_variance}")

        for client_index, (client, fit_res) in enumerate(results):
            local_loss = fit_res.metrics.get('loss', 0)

            group_id = next(group for group, client_indexes in G_ACTIVITY.items() if client_index in client_indexes)
            group_mean_loss = self.loss_avg_per_group[group_id]

            fairness_loss = self.fairness_regularization(server_round, client_index, local_loss, group_mean_loss, global_groups_variance)
            fit_res.metrics['loss'] = fairness_loss

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics.get('loss', 0)) for _, fit_res in results
        ]

        def aggregate(results: List[Tuple[NDArrays, float]]) -> NDArrays:
            loss_total = sum(loss for (_, loss) in results)

            weighted_weights = [
                [layer * loss for layer in weights] for weights, loss in results
            ]

            weights_prime: NDArrays = [
                reduce(np.add, layer_updates) / loss_total
                for layer_updates in zip(*weighted_weights)
            ]
            return weights_prime

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        metrics_aggregated = {}
        for client_index, (client, fit_res) in enumerate(results):
            loss = fit_res.metrics.get('loss', 0)
            weight = loss / total_loss
            num_examples = fit_res.num_examples
            print(f"Cliente {client.cid}: Perda = {loss}, Peso = {weight}, Exemplos = {num_examples}")
            self.all_losses.append(loss)
            self.all_weights.append(weight)
        
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        metrics_aggregated = {}
        loss_aggregated = None
        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = Net(NUM_CLIENTS, 1000).to(DEVICE)
        set_parameters(net, parameters_to_ndarrays(parameters))
        loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(net, testloader, server=True)

        metrics = {
            "rmse": rmse,
            "accuracy": accuracy,
            "precision_at_10": precision_at_10,
            "recall_at_10": recall_at_10,
            "RgrpActivity": RgrpActivity,
            "RgrpGender": RgrpGender,
            "RgrpAge": RgrpAge,
            "RgrpActivity_Losses": RgrpActivity_Losses,
            "RgrpGender_Losses": RgrpGender_Losses,
            "RgrpAge_Losses": RgrpAge_Losses
        }

        # Ajustar a taxa de aprendizado com base na nova RgrpActivity
        # self.evaluate_and_adjust_learning_rate(metrics["RgrpActivity"])

        print(f"Server-side evaluation :: Round {server_round}")
        print(f"Loss {loss} / RMSE {rmse} / Accuracy {accuracy} / Precision@10 {precision_at_10} / Recall@10 {recall_at_10}")
        print(f"RgrpActivity {RgrpActivity} / RgrpGender {RgrpGender} / RgrpAge {RgrpAge}")
        print(f"RgrpActivity_Losses {RgrpActivity_Losses} / RgrpGender_Losses {RgrpGender_Losses} / RgrpAge_Losses {RgrpAge_Losses}")
        
        # Verificar se a configuração já existe na lista RESULTS
        config_exists = False
        for result in RESULTS:
            if (result['learning_rate'] == self.learning_rate):
                
                # Adicionar nova tupla (server_round, RgrpActivity) na lista RgrpActivity
                result['RgrpActivity'].append((server_round, RgrpActivity))
                config_exists = True
                break

        if not config_exists:
            RESULTS.append({
                'learning_rate': self.learning_rate,
                'RgrpActivity': [(server_round, RgrpActivity)]
            })

        return loss, metrics


    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

# Configuração dos recursos do cliente
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

import optuna
import flwr as fl

def objective(trial):
    # Sugestão de hiperparâmetros pelo Optuna
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1000)

    strategy = FedCustom(
        initial_learning_rate = learning_rate,
    )

    # Executar simulação Flower
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )

    loss = None
    RgrpActivity = None
    if RESULTS:
        last_result = RESULTS[-1]
        loss = last_result.get("RgrpActivity", -1)
        RgrpActivity = next(x[1] for x in loss if x[0] == NUM_ROUNDS)

    # nome_do_arquivo = 'triais.txt'
    # with open(nome_do_arquivo, 'a') as arquivo:
    #     arquivo.write(f"trial: {trial}\n")
    #     arquivo.write(f"RESULTS: {RESULTS}\")\n")
    #     arquivo.write(f"RgrpActivity: {RgrpActivity}\n")
    # print(f"O código foi escrito no arquivo {nome_do_arquivo}")

    # O Optuna irá tentar minimizar o valor de retorno
    return RgrpActivity

# Criar o estudo de otimização
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=80)  # Define o número de iterações (trials)

# Imprimir os melhores parâmetros e o valor da função objetivo
best_params = study.best_params
best_value = study.best_value
print(f"Melhores parâmetros encontrados: {best_params}")
print(f"Melhor valor da função objetivo: {best_value}")

# Melhores parâmetros encontrados: {'learning_rate': 887.848686910475}
# Melhor valor da função objetivo: 0.00014227850344369303