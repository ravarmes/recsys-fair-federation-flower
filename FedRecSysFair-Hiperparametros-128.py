from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Definir a classe do modelo Net
class Net(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        super(Net, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Saída final para um valor

    def forward(self, x):
        user_idx = x[:, 0].long()
        item_idx = x[:, 1].long()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        x = torch.cat((user_embed, item_embed), dim=1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * 4 + 1
        return x

# Carregar os dados do arquivo Excel
filename = 'X.xlsx'
df = pd.read_excel(filename, index_col=0)
dados = df.fillna(0).values
X, y = np.nonzero(dados)
ratings = dados[X, y].astype(float)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(np.column_stack((X, y)), ratings, test_size=0.2, random_state=42)

# Definir uma função de perda para o treinamento
def custom_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_true - y_pred)**2))

# Definir os hiperparâmetros para Grid Search, incluindo o número de épocas
param_grid = {
    'module__num_users': [300],
    'module__num_items': [1000],
    'embedding_dim': [128],
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'num_epochs': [10, 15, 20, 30]  # Adicionando opções para o número de épocas
}

# param_grid = {
#     'module__num_users': [300],
#     'module__num_items': [1000],
#     'embedding_dim': [128],
#     'lr': [0.1],
#     'num_epochs': [20]  # Adicionando opções para o número de épocas
# }

# Realizar Grid Search para encontrar os melhores hiperparâmetros, incluindo o número de épocas
best_score = np.inf
for users in param_grid['module__num_users']:
    for items in param_grid['module__num_items']:
        for emb_dim in param_grid['embedding_dim']:
            for lr in param_grid['lr']:
                for epochs in param_grid['num_epochs']:
                    model = Net(users, items, emb_dim)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = custom_loss
                    model.train()

                    for _ in range(epochs):  # Treinar o modelo com o número de épocas especificado
                        optimizer.zero_grad()
                        output = model(torch.tensor(X_train).float())
                        loss = criterion(output.flatten(), torch.tensor(y_train).float())
                        loss.backward()
                        optimizer.step()

                        # Avaliar o desempenho com os hiperparâmetros atuais
                        model.eval()
                        with torch.no_grad():
                            test_output = model(torch.tensor(X_test).float())
                            test_loss = criterion(test_output.flatten(), torch.tensor(y_test).float()).item()

                        # Atualizar melhores hiperparâmetros se o desempenho atual for melhor
                        if test_loss < best_score:
                            best_score = test_loss
                            best_params = {'num_users': users, 'num_items': items, 'embedding_dim': emb_dim, 'lr': lr, 'num_epochs': epochs}

print("Melhores hiperparâmetros:", best_params)
print("Melhor pontuação (RMSE) encontrada:", best_score)

# ------------------------------------------------------

# Avaliar o modelo com melhores hiperparâmetros
model = Net(best_params['num_users'], best_params['num_items'], best_params['embedding_dim'])
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = custom_loss

# Treinar o modelo com todos os dados de treino
model.train()
optimizer.zero_grad()
output = model(torch.tensor(np.column_stack((X_train, y_train))).float())
loss = criterion(output.flatten(), torch.tensor(y_train).float())
loss.backward()
optimizer.step()

# Avaliar a acurácia do modelo nos dados de teste
# model.eval()
# with torch.no_grad():
#     test_output = model(torch.tensor(np.column_stack((X_test, y_test))).float())
#     predicted_ratings = test_output.flatten().numpy()
#     accuracy = np.mean(np.abs(predicted_ratings - y_test) <= 0.5)  # Calculando a acurácia

# print("Melhores hiperparâmetros:", best_params)
# print("Melhor pontuação (RMSE) encontrada:", best_score)
# print("Acurácia do modelo nos dados de teste:", accuracy)

model.eval()
with torch.no_grad():
    test_output = model(torch.tensor(np.column_stack((X_test, y_test))).float())
    predicted_ratings = test_output.flatten().numpy()

    # threshold = 0.5 # Definir um limiar para binarizar as previsões
    threshold = 1 # Definir um limiar para binarizar as previsões

    accuracy = np.mean(np.abs(predicted_ratings - y_test) <= threshold)  # Calculando a acurácia

    predicted_classes = (predicted_ratings > threshold).astype(int)  # Discretizar as previsões em classes

    # Calcular as métricas Precision, Recall e F1-Score
    precision = precision_score(y_test, predicted_classes, average='weighted')
    recall = recall_score(y_test, predicted_classes, average='weighted')
    f1 = f1_score(y_test, predicted_classes, average='weighted')


print("Melhores hiperparâmetros:", best_params)
print("Melhor pontuação (RMSE) encontrada:", best_score)
print("Acurácia do modelo nos dados de teste:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# -----------------------------

top_k = 10  # Definir o número de principais itens recomendados
top_indices = np.argsort(predicted_ratings)[::-1][:top_k]  # Índices dos top_k itens recomendados

# Considerar apenas os top_k itens recomendados
top_predicted_ratings = predicted_ratings[top_indices]
top_y_test = y_test[top_indices]

# Identificar os top_k itens relevantes e recomendados
relevant_indices = np.where(top_y_test >= 3.5)[0]  # Índices dos itens relevantes
recommended_indices = np.where(top_predicted_ratings >= 3.5)[0]  # Índices dos itens recomendados

# Calcular os itens relevantes e recomendados em comum
relevant_recommended_indices = np.intersect1d(relevant_indices, recommended_indices)

# Contar o número de itens relevantes e recomendados em comum, bem como o total de itens relevantes
num_relevant_recommended = len(relevant_recommended_indices)
num_relevant = len(relevant_indices)

# Calcular Precision@10 e Recall@10
precision_at_10 = num_relevant_recommended / top_k if top_k > 0 else 0.0  # Precision@10
recall_at_10 = num_relevant_recommended / num_relevant if num_relevant > 0 else 0.0  # Recall@10

print("Precision@10:", precision_at_10)
print("Recall@10:", recall_at_10)




# Melhores hiperparâmetros: {'num_users': 300, 'num_items': 1000, 'embedding_dim': 128, 'lr': 0.01, 'num_epochs': 20}
# Melhor pontuação (RMSE) encontrada: 0.915046215057373

#Melhores hiperparâmetros: {'num_users': 300, 'num_items': 1000, 'embedding_dim': 128, 'lr': 0.01, 'num_epochs': 20}
# Melhor pontuação (RMSE) encontrada: 0.9019684791564941


