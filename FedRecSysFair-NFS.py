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
from torch.utils.data import ConcatDataset



import flwr as fl

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


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
    testloader_data = []

    for round_num in range(24):  # Iterar sobre os 24 rounds
        round_train_loader = []

        for cliente_id in range(num_clients):
            if cliente_id <= 14:
                num_avaliacoes = 32
            else:
                num_avaliacoes = 16

            dados_cliente = np.array(cliente_avaliacoes[cliente_id])
            np.random.shuffle(dados_cliente)  # Embaralhar as avaliações do cliente
            X_train = dados_cliente[:num_avaliacoes, :2]
            y_train = dados_cliente[:num_avaliacoes, 2]

            dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            round_train_loader.append(dataset)

            # Adicionar dados de teste do cliente à lista de testes
            testloader_data.extend(dados_cliente)

        concatenated_dataset = ConcatDataset(round_train_loader)
        batch_size = 32*15 + 16*(num_clients - 15)  # Definir o batch_size conforme especificado

        train_loader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)
        trainloaders.append(train_loader)

    # Construir o testloader com 10% dos dados acumulados
    num_test_samples = len(testloader_data)
    num_test_samples_10_percent = int(0.1 * num_test_samples)
    random.shuffle(testloader_data)
    X_test_all = np.array([x[:2] for x in testloader_data[:num_test_samples_10_percent]])
    y_test_all = np.array([x[2] for x in testloader_data[:num_test_samples_10_percent]])
    test_dataset = TensorDataset(torch.from_numpy(X_test_all).float(), torch.from_numpy(y_test_all).float())
    testloader = DataLoader(test_dataset, batch_size=(32*15 + 16*(num_clients - 15)), shuffle=True)


    return df, trainloaders, [], testloader



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
    

def train(net, trainloader, epochs: int, lotes_por_rodada: int, learning_rate : float):
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
        # print(f"[NFS] Número de lotes processados: {num_batches}")
        # print(f"[NFS] Número de exemplos processados: {num_examples}")
        # print(f"[NFS] Época {epoch + 1}: loss {epoch_loss}")
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
        #precision_at_10, recall_at_10 = calculate_f1_recall_at_k(outputs, target, k=10, threshold=3.5)
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
    
    # print("recomendacoes_df")
    # print(recomendacoes_df)

    return RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses


def evaluate(net, testloader, tolerance=0.7, server=True):
    loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = test(net, testloader, server=True)
    metrics = {"rmse": rmse, "accuracy": accuracy, "precision_at_10": precision_at_10, "recall_at_10": recall_at_10, "RgrpActivity": RgrpActivity, "RgrpGender": RgrpGender, "RgrpAge": RgrpAge, "RgrpActivity_Losses": RgrpActivity_Losses, "RgrpGender_Losses": RgrpGender_Losses, "RgrpAge_Losses": RgrpAge_Losses}  # Agrupar RMSE e accuracy em um dicionário
    print(f"loss {loss} / RMSE {rmse} / accuracy {accuracy} / Precision@10 {precision_at_10} / Recall@10 {recall_at_10}")
    print(f"RgrpActivity {RgrpActivity} / RgrpGender {RgrpGender} / RgrpAge {RgrpAge}")
    print(f"RgrpActivity_Losses {RgrpActivity_Losses} / RgrpGender_Losses {RgrpGender_Losses} / RgrpAge_Losses {RgrpAge_Losses}")
    return loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses

avaliacoes_df, trainloaders, valloaders, testloader = load_datasets(num_clients=300, filename="X.xlsx")
results = []
l_loss = l_rmse = l_accuracy = l_precision_at_10 = l_recall_at_10 = l_RgrpActivity = l_RgrpGender = l_RgrpAge = l_RgrpActivity_Losses = l_RgrpGender_Losses = l_RgrpAge_Losses = []
net = Net(300, 1000).to(DEVICE)
for round in range (0, 25):
    print(f"ROUND [{round}]")
    trainloader = trainloaders[int(round)]
    #valloader = valloaders[int(round)]
    train(net=net, trainloader=trainloader, epochs=20, lotes_por_rodada=round, learning_rate=0.01)

    loss, rmse, accuracy, precision_at_10, recall_at_10, RgrpActivity, RgrpGender, RgrpAge, RgrpActivity_Losses, RgrpGender_Losses, RgrpAge_Losses = evaluate(net=net, testloader=testloader, tolerance=0.7, server=True)
    l_loss.append((round, loss))
    l_rmse.append((round, rmse))
    l_accuracy.append((round, accuracy))
    l_precision_at_10.append((round, precision_at_10))
    l_recall_at_10.append((round, recall_at_10))
    l_RgrpActivity.append((round, RgrpActivity))
    l_RgrpGender.append((round, RgrpGender))
    l_RgrpAge.append((round, RgrpAge))
    l_RgrpActivity_Losses.append((round, RgrpActivity_Losses))
    l_RgrpGender_Losses.append((round, RgrpGender_Losses))
    l_RgrpAge_Losses.append((round, RgrpAge_Losses))

    print(f"Server-side evaluation :: Round {round}")
    print(f"loss {loss} / RMSE {rmse} / accuracy {accuracy} / Precision@10 {precision_at_10} / Recall@10 {recall_at_10}")
    print(f"RgrpActivity {RgrpActivity} / RgrpGender {RgrpGender} / RgrpAge {RgrpAge}")
    print(f"RgrpActivity_Losses {RgrpActivity_Losses} / RgrpGender_Losses {RgrpGender_Losses} / RgrpAge_Losses {RgrpAge_Losses}")

metrics = {"rmse": l_rmse, "accuracy": l_accuracy, "precision_at_10": l_precision_at_10, "recall_at_10": l_recall_at_10, "RgrpActivity": l_RgrpActivity, "RgrpGender": l_RgrpGender, "RgrpAge": l_RgrpAge, "RgrpActivity_Losses": l_RgrpActivity_Losses, "RgrpGender_Losses": l_RgrpGender_Losses, "RgrpAge_Losses": l_RgrpAge_Losses}

# print("\n\nRESUMO")
# print(metrics)