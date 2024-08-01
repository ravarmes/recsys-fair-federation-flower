# !pip install -q flwr[simulation] torch torchvision

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

# Configuração do dispositivo
DEVICE = torch.device("cpu")  # Tente "cuda" para treinar na GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

NUM_CLIENTS = 300

def verificar_trainloaders(trainloaders):
    for i, trainloader in enumerate(trainloaders):
        if i == 0 or i == 299:
            print(f"Trainloader {i+1} (Cliente {i+1}):")
            for data in trainloader:
                inputs, labels = data
                print("Inputs (Usuário, Item):", inputs)
                print("Labels (Avaliações):", labels)
                print()  # Linha em branco para melhor visualização
            print("============== Fim do DataLoader ============")
            print()  # Linha extra para separar cada DataLoader

def set_random_seed(seed: int):
    """Função para configurar as sementes para reprodutibilidade."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para GPUs com múltiplas GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets(num_clients: int, filename: str, seed: int = 42):
    # Configura a semente para garantir reprodutibilidade
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

        batch_size = 32 if cliente_id <= 14 else 16  # Configurando o tamanho do lote
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        
        # Adicionar dados de cada cliente para seleção de teste
        testloader_data.extend(dados_cliente)

    # Usar 10% dos dados acumulados para o teste
    num_test_samples = len(testloader_data)
    
    # Seleção de teste com semente
    random.seed(seed)
    test_data_sample = random.sample(testloader_data, num_test_samples // 8)  # 10% dos dados para teste

    # Separar dados de teste para evitar sobreposição
    test_data_set = set(map(tuple, test_data_sample))
    
    # Modificação para acessar os tensor do dataset original
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

# Carregar dados e garantir reprodutibilidade
avaliacoes_df, trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, filename="X.xlsx")

class Net(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

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
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, cid, epochs: int, lotes_por_rodada: int, learning_rate : float):
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
    recomendacoes_df = net.predict_all(300, 1000)
    omega = ~avaliacoes_df.isnull()

    G_ACTIVITY = {1: list(range(0, 15)), 2: list(range(15, 300))}
    G_GENDER = {1: [...], 2: [...]}  # Manter seus grupos de gênero
    G_AGE = {1: [...], 2: [...]}  # Manter seus grupos de idade

    glv = GroupLossVariance(avaliacoes_df, omega, G_ACTIVITY, 1)
    RgrpActivity = glv.evaluate(recomendacoes_df)
    RgrpActivity_Losses = glv.get_losses(recomendacoes_df)
    
    glv = GroupLossVariance(avaliacoes_df, omega, G_GENDER, 1)
    RgrpGender = glv.evaluate(recomendacoes_df)
    RgrpGender_Losses = glv.get_losses(recomendacoes_df)
    
    glv = GroupLossVariance(avaliacoes_df, omega, G_AGE, 1)
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
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        learning_rate = config["learning_rate"]
        lotes_por_rodada = config["lotes_por_rodada"]

        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        num_examples, loss = train(self.net, self.trainloader, self.cid, epochs=local_epochs, lotes_por_rodada=lotes_por_rodada, learning_rate=learning_rate)

        metrics = {
            "group": 42,
            "li": 3.14,
            "loss": loss,
        }

        result = get_parameters(self.net), num_examples, metrics
        return result

    def evaluate(self, parameters, config):
        print(f"[Cliente {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(self.net, self.valloader, server=False)
        return float(loss), len(self.valloader), {"rmse": float(rmse), "accuracy": float(accuracy)}

def client_fn(cid) -> FlowerClient:
    print(f"client_fn({cid})")
    net = Net(300, 1000).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    flower_client = FlowerClient(cid, net, trainloader, valloader)
    return flower_client.to_client()

def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net(300, 1000).to(DEVICE)
    set_parameters(net, parameters)
    loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(net, testloader, server=True)
    metrics = {"rmse": rmse, "accuracy": accuracy, "precision_at_10": precision_at_10, "recall_at_10": recall_at_10, 
               "RgrpActivity": RgrpActivity, "RgrpGender": RgrpGender, "RgrpAge": RgrpAge, 
               "RgrpActivity_Losses": RgrpActivity_Losses, "RgrpGender_Losses": RgrpGender_Losses, "RgrpAge_Losses": RgrpAge_Losses}
    print(f"Server-side evaluation :: Round {server_round}")
    print(f"loss {loss} / RMSE {rmse} / accuracy {accuracy} / Precision@10 {precision_at_10} / Recall@10 {recall_at_10}")
    print(f"RgrpActivity {RgrpActivity} / RgrpGender {RgrpGender} / RgrpAge {RgrpAge}")
    print(f"RgrpActivity_Losses {RgrpActivity_Losses} / RgrpGender_Losses {RgrpGender_Losses} / RgrpAge_Losses {RgrpAge_Losses}")
    return loss, metrics

def fit_config(server_round: int):
    set_random_seed(42)  # Configuração da semente
    config = {
        "server_round": server_round,
        "local_epochs": 20,
        "lotes_por_rodada": server_round,
        "learning_rate": 0.01,
    }
    return config

# Inicialize o modelo global
set_random_seed(42)  # Garantir reprodutibilidade antes da inicialização

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net(300, 1000))),
    evaluate_fn=evaluate,
    on_fit_config_fn=fit_config,
)

# Especifique os recursos do cliente se precisar de GPU
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),  # Defina o número de rodadas
    strategy=strategy,
    client_resources=client_resources,
)
