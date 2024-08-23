import random as random
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import pandas as pd
import numpy as np

class AlgorithmImpartiality():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
    
    def evaluate(self, X_est, n):
        list_X_est = []
        for x in range(0, n):
            print("x: ", x)
            list_X_est.append(self.get_X_est4(X_est.copy()))
        return list_X_est
    
    def evaluate_federated(self, X_est, list_dif_mean, list_dif_var, n):
        list_X_est = []
        for x in range(0, n):
            print("x: ", x)
            list_X_est.append(self.get_X_est2_federated(X_est.copy(), list_dif_mean, list_dif_var))
        return list_X_est
        
    def get_differences_means(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)

        E = (X_est - X)
        losses = E.mean(axis=self.axis)
        return losses

    def get_differences_vars(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    # Estratégia de perturbação baseada na MÉDIA das diferenças de avaliações
    def get_X_est(self, X_est):
        list_dif_mean = (self.get_differences_means(X_est)).tolist()
        for i in range(0, len(X_est.index)):
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, list_dif_mean[i])
                else:
                    value = X_est.iloc[i, j] + random.uniform(list_dif_mean[i], 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est

    # Estratégia de perturbação baseada na variância das diferenças de avaliações
    #Variância máxima = 16 (ou seja, 4*4)
    #Variância normalizada: li/4
    def get_X_est2(self, X_est):
        list_dif_mean = (self.get_differences_means(X_est)).tolist()
        list_dif_var = (self.get_differences_vars(X_est)).tolist()
        #print("list_dif_mean: ", list_dif_mean)
        #print("list_dif_var: ", list_dif_var)
        for i in range(0, len(X_est.index)):
            #var_norm = 16.0*list_dif_var[i]/4.0
            var_norm = list_dif_var[i]/4.0
            #print("i: ", i, " - var_norm: ", var_norm)
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, var_norm)
                else:
                    value = X_est.iloc[i, j] + random.uniform(var_norm, 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est
    
    # Estratégia 2 otimizada
    def get_X_est3(self, X_est):
        list_dif_mean = self.get_differences_means(X_est)
        list_dif_var = self.get_differences_vars(X_est) / 4.0
        # Convertendo para arrays NumPy para operação vetorizada
        list_dif_mean = list_dif_mean.to_numpy()
        list_dif_var = list_dif_var.to_numpy()
        # Cria uma matriz de valores aleatórios com base nas diferenças de variância
        random_values = np.random.uniform(0, 1, X_est.shape) * list_dif_var[:, np.newaxis]
        random_values = np.where(list_dif_mean[:, np.newaxis] > 0, random_values, -random_values)
        # Atualiza X_est com valores aleatórios, garantindo que os valores estejam entre 1 e 5
        X_est += random_values
        X_est = X_est.clip(lower=1, upper=5)
        return X_est
    
    # Estratégia 2 otimizada (outra versão otimizada)
    def get_X_est4(self, X_est):
        list_dif_mean = self.get_differences_means(X_est)
        list_dif_var = self.get_differences_vars(X_est) / 4.0

        print("\nAlgorithmImpartiality :: get_X_est4")
        print("(self.get_differences_means(X_est)).tolist()")
        print((self.get_differences_means(X_est)).tolist())
        print("(self.get_differences_vars(X_est)).tolist()")
        print((self.get_differences_vars(X_est)).tolist())

        # Convertendo para arrays NumPy para operação vetorizada
        list_dif_mean = list_dif_mean.to_numpy()
        list_dif_var = list_dif_var.to_numpy()

        # Determinar os limites inferior e superior para a geração de valores aleatórios
        lower_bounds = np.where(list_dif_mean[:, np.newaxis] > 0, 0, list_dif_var[:, np.newaxis])
        upper_bounds = np.where(list_dif_mean[:, np.newaxis] > 0, list_dif_var[:, np.newaxis], 0)
        
        # Ajustando o caso onde o limite inferior é maior que o superior
        # Isso aconteceria em casos onde list_dif_var[i] é negativo, o que não deveria acontecer nesta lógica, mas é um ajuste de segurança
        lower_bounds, upper_bounds = np.minimum(lower_bounds, upper_bounds), np.maximum(lower_bounds, upper_bounds)

        lower_bounds = lower_bounds.astype('float')
        upper_bounds = upper_bounds.astype('float')
        
        # Cria uma matriz de valores aleatórios dentro dos limites definidos
        random_values = np.random.uniform(lower_bounds, upper_bounds, X_est.shape)
        
        # Atualiza X_est com valores aleatórios, garantindo que os valores estejam entre 1 e 5
        X_est = X_est.add(random_values)
        X_est = X_est.clip(lower=1, upper=5)
        return X_est
    
    def get_X_est2_federated(self, X_est, list_dif_mean, list_dif_var):
        # # print("list_dif_mean: ", list_dif_mean)
        # # print("list_dif_var: ", list_dif_var)
        # print(f"get_X_est2_federated :: len(X_est.index) {len(X_est.index)}")
        # print(f"get_X_est2_federated :: len(X_est.columns) {len(X_est.columns)}")
        for i in range(0, len(X_est.index)):
            #var_norm = 16.0*list_dif_var[i]/4.0
            var_norm = list_dif_var[i]/4.0
            #print("i: ", i, " - var_norm: ", var_norm)
            for j in range(0, len(X_est.columns)):
                if (list_dif_mean[i] > 0):
                    value = X_est.iloc[i, j] + random.uniform(0, var_norm)
                else:
                    value = X_est.iloc[i, j] + random.uniform(var_norm, 0)
                value = 1 if value < 1 else value
                value = 5 if value > 5 else value
                X_est.iloc[i, j] = value
        return X_est
    
    # Estratégia 2 otimizada (outra versão otimizada)
    def get_X_est4_federated(self, X_est, list_dif_mean, list_dif_var):
        list_dif_mean = self.get_differences_means(X_est)
        list_dif_var = self.get_differences_vars(X_est) / 4.0
        # Convertendo para arrays NumPy para operação vetorizada
        list_dif_mean = list_dif_mean.to_numpy()
        list_dif_var = list_dif_var.to_numpy()
        
        # Determinar os limites inferior e superior para a geração de valores aleatórios
        lower_bounds = np.where(list_dif_mean[:, np.newaxis] > 0, 0, list_dif_var[:, np.newaxis])
        upper_bounds = np.where(list_dif_mean[:, np.newaxis] > 0, list_dif_var[:, np.newaxis], 0)

        lower_bounds = lower_bounds.astype('float')
        upper_bounds = upper_bounds.astype('float')
        
        # Ajustando o caso onde o limite inferior é maior que o superior
        # Isso aconteceria em casos onde list_dif_var[i] é negativo, o que não deveria acontecer nesta lógica, mas é um ajuste de segurança
        lower_bounds, upper_bounds = np.minimum(lower_bounds, upper_bounds), np.maximum(lower_bounds, upper_bounds)

        # Cria uma matriz de valores aleatórios dentro dos limites definidos
        random_values = np.random.uniform(lower_bounds, upper_bounds, X_est.shape)
        
        # Atualiza X_est com valores aleatórios, garantindo que os valores estejam entre 1 e 5
        X_est = X_est.add(random_values)
        X_est = X_est.clip(lower=1, upper=5)
        return X_est
    
    def losses_to_Z(list_losses):
        Z = []
        linha = []
        for i in range (0, len(list_losses[0])):
            for losses in list_losses:
                linha.append(losses.values[i])
            Z.append(linha.copy())
            linha.clear()
        return Z

    def matrices_Zs(Z, G): # return a Z matrix for each group
        list_Zs = []
        for group in G: # G = {1: [1,2], 2: [3,4,5]}
            Z_ = []
            list_users = G[group]
            for user in list_users:
                Z_.append(Z[user].copy())   
            list_Zs.append(Z_)
        return list_Zs
    

    # def make_matrix_X_gurobi(list_X_est, G, list_Zs):
    #     m = gp.Model()
        
    #     # Identificar todos os usuários e suas recomendações correspondentes
    #     all_users = sorted(set(user for users in G.values() for user in users))
    #     n_recommendations = len(list_X_est)
        
    #     # Criar variáveis de decisão para cada combinação de usuário e recomendação
    #     x = m.addVars(all_users, list(range(n_recommendations)), vtype=gp.GRB.BINARY, name="x")
        
    #     # Criar as preferências baseadas em list_Zs
    #     # Supondo que list_Zs seja um lista de arrays onde cada elemento corresponde aos usuários em G
    #     preferences = {(user, r): list_Zs[group_idx][user_idx][r] 
    #                 for group_idx, users in enumerate(G.values()) 
    #                 for user_idx, user in enumerate(users) 
    #                 for r in range(n_recommendations)}
        
    #     # Adicionar a função objetivo: minimizar a variação das injustiças entre os grupos
    #     group_losses = [gp.quicksum(preferences[user, r] * x[user, r] for user in users for r in range(n_recommendations)) / len(users) 
    #                     for users in G.values()]
    #     avg_loss = gp.quicksum(group_losses) / len(G)
    #     m.setObjective(gp.quicksum((loss - avg_loss) * (loss - avg_loss) for loss in group_losses), gp.GRB.MINIMIZE)
        
    #     # Restrição: garantir que cada usuário receba exatamente uma recomendação
    #     m.addConstrs((x.sum(user, '*') == 1 for user in all_users), "OneRecommendationPerUser")
        
    #     # Otimizar o modelo
    #     m.optimize()
        
    #     # Coletar as decisões de recomendação para cada usuário
    #     decision = m.getAttr('X', x)
    #     # Reconstruir a matriz de recomendações finais
    #     # Inicializando a matriz X_gurobi com zeros
    #     X_gurobi = pd.DataFrame(0, index=all_users, columns=list_X_est[0].columns)

    #     # Preenchendo X_gurobi com as recomendações selecionadas
    #     for user in all_users:
    #         for r in range(n_recommendations):
    #             if decision[user, r] > 0.5:  # Se a recomendação foi escolhida para o usuário
    #                 # Adicionando as recomendações à linha correspondente do usuário em X_gurobi
    #                 # Supõe-se que cada list_X_est[r] é um DataFrame com a mesma estrutura de colunas que X_gurobi
    #                 X_gurobi.loc[user] = list_X_est[r].loc[user]
        
    #     return X_gurobi
    
    
    # versão inicial (não refatorada - apenas para dois grupos)
    def make_matrix_X_gurobi(list_X_est, G, list_Zs):

        # User labels and Rindv (individual injustices)
        users_g1 = ["U{}".format(user) for user in G[1]]  
        ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g2 = ["U{}".format(user) for user in G[2]] 
        ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 

        # Dictionary with individual losses
        Z1 = list_Zs[0]
        preferencias1 = dict()
        for i, user in enumerate(users_g1):
            for j, l in enumerate(ls_g1):
                preferencias1[user, l] = Z1[i][j]

        Z2 = list_Zs[1]
        preferencias2 = dict()
        for i, user in enumerate(users_g2):
            for j, l in enumerate(ls_g2):  
                preferencias2[user, l] = Z2[i][j]


        # Initialize the model
        m = gp.Model()

        # Decision variables
        x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
        x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)


        # Objective function
        # In this case, the objective function seeks to minimize the variance between the injustices of the groupS (Li) 
        # Li can also be understood as the average of the individual injustices of group i. 
        # Rgrp: the variance of all the injustices of the groups (Li).

        L1 = x1.prod(preferencias1)/len(users_g1)
        L2 = x2.prod(preferencias2)/len(users_g2)
        LMean = (L1 + L2) / 2
        Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2 )/2

        m.setObjective( Rgrp , sense=gp.GRB.MINIMIZE)

        # Restrictions that ensure all users will have a Rindv (individual injustice calculation)
        c1 = m.addConstrs(x1.sum(i, '*') == 1 for i in users_g1)
        c2 = m.addConstrs(x2.sum(i, '*') == 1 for i in users_g2)

        # Run the model
        m.optimize()

        matrix_final = []
        indices = list_X_est[0].index.tolist()
        qtd_g1 = 0
        qtd_g2 = 0

        for i in indices:
            if i in G[1]:
                for j in ls_g1:
                    if round(x1['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g1 = qtd_g1 + 1
            if i in G[2]:
                for j in ls_g2:
                    if round(x2['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g2 = qtd_g2 + 1

        X_gurobi = pd.DataFrame (matrix_final, columns=list_X_est[0].columns.values)
        X_gurobi['UserID'] = indices
        X_gurobi.set_index("UserID", inplace = True)
        
        return X_gurobi


    # # versão que tenta diminuir também o RMSE (incluindo a medida na função objetivo)
    # def make_matrix_X_gurobi(X, list_X_est, G, list_Zs):
    #     m = gp.Model("optimization")

    #     # Preparando os índices dos usuários e recomendações
    #     all_users = sorted(set(user for users in G.values() for user in users))
    #     n_recommendations = len(list_X_est)

    #     # Criando variáveis de decisão
    #     x = m.addVars(all_users, range(n_recommendations), vtype=gp.GRB.BINARY, name="x")

    #     # Supondo que list_Zs seja uma lista de arrays, cada um representando perdas para um grupo específico
    #     preferences = {}
    #     for group_idx, users in G.items():
    #         # Acessando as perdas do grupo específico
    #         losses = list_Zs[group_idx - 1]  # Ajustando o índice baseado em 0
    #         for user_idx, user in enumerate(users):
    #             for rec_idx in range(n_recommendations):
    #                 # Supondo que cada array em list_Zs tem a forma [n_usuários_no_grupo, n_recomendações]
    #                 # e está alinhado com a ordem dos usuários em G
    #                 preferences[(user, rec_idx)] = losses[user_idx, rec_idx]

    #     # Definindo a função objetivo: aqui simplificada para focar na minimização da injustiça do grupo
    #     m.setObjective(gp.quicksum(preferences[user, rec] * x[user, rec] for user in all_users for rec in range(n_recommendations)), gp.GRB.MINIMIZE)

    #     # Adicionando restrições: cada usuário recebe exatamente uma recomendação
    #     m.addConstrs((x.sum(user, '*') == 1 for user in all_users), name="one_recommendation")

    #     # Otimiza o modelo
    #     m.optimize()

    #     # Reconstrução da matriz de recomendações baseada nas decisões de otimização
    #     X_gurobi = pd.DataFrame(0, index=all_users, columns=X.columns)
    #     for user in all_users:
    #         for rec in range(n_recommendations):
    #             if x[user, rec].X > 0.5:  # Se a recomendação foi escolhida
    #                 X_gurobi.loc[user] += list_X_est[rec].loc[user]

    #     # Preenchendo valores faltantes, se necessário
    #     X_gurobi.fillna(0, inplace=True)

    #     return X_gurobi


    # def make_matrix_X_gurobi_seven_groups(list_X_est, G, list_Zs):
    #     # Inicializando o modelo
    #     m = gp.Model()

    #     # Variáveis de decisão para cada grupo
    #     decision_vars = {}
    #     preferences = {}
    #     for group_idx in range(1, 8):  # 7 grupos
    #         users_group = ["U{}".format(user) for user in G[group_idx]]
    #         ls_group = ["l{}".format(j + 1) for j in range(len(list_X_est))]
    #         decision_vars[group_idx] = m.addVars(users_group, ls_group, vtype=gp.GRB.BINARY)

    #         # Dicionário com as perdas individuais
    #         Z_group = list_Zs[group_idx-1]  # Ajuste para indexação base-0
    #         for i, user in enumerate(users_group):
    #             for j, l in enumerate(ls_group):
    #                 preferences[(user, l)] = Z_group[i][j]

    #     # Função objetivo
    #     L = []
    #     for group_idx in range(1, 8):
    #         L_group = decision_vars[group_idx].prod(preferences) / len(G[group_idx])
    #         L.append(L_group)

    #     LMean = sum(L) / 7
    #     Rgrp = sum((L_group - LMean)**2 for L_group in L) / 7
    #     m.setObjective(Rgrp, gp.GRB.MINIMIZE)

    #     # Restrições: garantir que cada usuário receba exatamente uma recomendação
    #     for group_vars in decision_vars.values():
    #         m.addConstrs(group_vars.sum(user, '*') == 1 for user in group_vars.keys())

    #     # Otimizar o modelo
    #     m.optimize()

    #     # Reconstruir a matriz de recomendações finais
    #     matrix_final = []
    #     indices = list_X_est[0].index.tolist()

    #     # Coletar as decisões de recomendação para cada usuário em todos os grupos
    #     for group_idx in range(1, 8):
    #         group_decision = decision_vars[group_idx]
    #         for user in indices:
    #             user_var = "U{}".format(user)
    #             if user_var in group_decision:
    #                 for rec_idx, var in group_decision.select(user_var, '*'):
    #                     if round(var.X) == 1:
    #                         # Extrai o índice da recomendação selecionada
    #                         rec_num = int(rec_idx.split('l')[1]) - 1
    #                         matrix = list_X_est[rec_num].loc[user]
    #                         matrix_final.append(matrix.tolist())

    #     # Construir X_gurobi como DataFrame
    #     X_gurobi = pd.DataFrame(matrix_final, columns=list_X_est[0].columns.values)
    #     X_gurobi['UserID'] = indices
    #     X_gurobi.set_index('UserID', inplace=True)

    #     return X_gurobi


    # versão inicial (não refatorada - apenas para dois grupos)
    def make_matrix_X_gurobi_seven_groups(list_X_est, G, list_Zs):

        # User labels and Rindv (individual injustices)
        users_g1 = ["U{}".format(user) for user in G[1]]  
        ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g2 = ["U{}".format(user) for user in G[2]] 
        ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g3 = ["U{}".format(user) for user in G[3]] 
        ls_g3 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g4 = ["U{}".format(user) for user in G[4]] 
        ls_g4 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g5 = ["U{}".format(user) for user in G[5]] 
        ls_g5 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g6 = ["U{}".format(user) for user in G[6]] 
        ls_g6 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g7 = ["U{}".format(user) for user in G[7]] 
        ls_g7 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 

        # Dictionary with individual losses
        Z1 = list_Zs[0]
        preferencias1 = dict()
        for i, user in enumerate(users_g1):
            for j, l in enumerate(ls_g1):
                preferencias1[user, l] = Z1[i][j]

        Z2 = list_Zs[1]
        preferencias2 = dict()
        for i, user in enumerate(users_g2):
            for j, l in enumerate(ls_g2):  
                preferencias2[user, l] = Z2[i][j]

        Z3 = list_Zs[2]
        preferencias3 = dict()
        for i, user in enumerate(users_g3):
            for j, l in enumerate(ls_g3):  
                preferencias3[user, l] = Z3[i][j]

        Z4 = list_Zs[3]
        preferencias4 = dict()
        for i, user in enumerate(users_g4):
            for j, l in enumerate(ls_g4):  
                preferencias4[user, l] = Z4[i][j]

        Z5 = list_Zs[4]
        preferencias5 = dict()
        for i, user in enumerate(users_g5):
            for j, l in enumerate(ls_g5):  
                preferencias5[user, l] = Z5[i][j]

        Z6 = list_Zs[5]
        preferencias6 = dict()
        for i, user in enumerate(users_g6):
            for j, l in enumerate(ls_g6):  
                preferencias6[user, l] = Z6[i][j]

        Z7 = list_Zs[6]
        preferencias7 = dict()
        for i, user in enumerate(users_g7):
            for j, l in enumerate(ls_g7):  
                preferencias7[user, l] = Z7[i][j]


        # Initialize the model
        m = gp.Model()

        # Decision variables
        x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
        x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)
        x3 = m.addVars(users_g3, ls_g3, vtype=gp.GRB.BINARY)
        x4 = m.addVars(users_g4, ls_g4, vtype=gp.GRB.BINARY)
        x5 = m.addVars(users_g5, ls_g5, vtype=gp.GRB.BINARY)
        x6 = m.addVars(users_g6, ls_g6, vtype=gp.GRB.BINARY)
        x7 = m.addVars(users_g7, ls_g7, vtype=gp.GRB.BINARY)


        # Objective function
        # In this case, the objective function seeks to minimize the variance between the injustices of the groupS (Li) 
        # Li can also be understood as the average of the individual injustices of group i. 
        # Rgrp: the variance of all the injustices of the groups (Li).

        L1 = x1.prod(preferencias1)/len(users_g1)
        L2 = x2.prod(preferencias2)/len(users_g2)
        L3 = x3.prod(preferencias3)/len(users_g3)
        L4 = x4.prod(preferencias4)/len(users_g4)
        L5 = x5.prod(preferencias5)/len(users_g5)
        L6 = x6.prod(preferencias6)/len(users_g6)
        L7 = x7.prod(preferencias7)/len(users_g7)
        LMean = (L1 + L2 + L3 + L4 + L5 + L6 + L7) / 7
        Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2 + (L3 - LMean)**2 + (L4 - LMean)**2 + (L5 - LMean)**2 + (L6 - LMean)**2 + (L7 - LMean)**2)/7

        m.setObjective( Rgrp , sense=gp.GRB.MINIMIZE)

        # Restrictions that ensure all users will have a Rindv (individual injustice calculation)
        c1 = m.addConstrs(x1.sum(i, '*') == 1 for i in users_g1)
        c2 = m.addConstrs(x2.sum(i, '*') == 1 for i in users_g2)
        c3 = m.addConstrs(x3.sum(i, '*') == 1 for i in users_g3)
        c4 = m.addConstrs(x4.sum(i, '*') == 1 for i in users_g4)
        c5 = m.addConstrs(x5.sum(i, '*') == 1 for i in users_g5)
        c6 = m.addConstrs(x6.sum(i, '*') == 1 for i in users_g6)
        c7 = m.addConstrs(x7.sum(i, '*') == 1 for i in users_g7)

        # Run the model
        m.optimize()

        matrix_final = []
        indices = list_X_est[0].index.tolist()
        qtd_g1 = 0
        qtd_g2 = 0
        qtd_g3 = 0
        qtd_g4 = 0
        qtd_g5 = 0
        qtd_g6 = 0
        qtd_g7 = 0

        for i in indices:
            if i in G[1]:
                for j in ls_g1:
                    if round(x1['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g1 = qtd_g1 + 1
            if i in G[2]:
                for j in ls_g2:
                    if round(x2['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g2 = qtd_g2 + 1
            if i in G[3]:
                for j in ls_g3:
                    if round(x3['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g3 = qtd_g3 + 1
            if i in G[4]:
                for j in ls_g4:
                    if round(x4['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g4 = qtd_g4 + 1
            if i in G[5]:
                for j in ls_g5:
                    if round(x5['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g5 = qtd_g5 + 1
            if i in G[6]:
                for j in ls_g6:
                    if round(x6['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g6 = qtd_g6 + 1
            if i in G[7]:
                for j in ls_g7:
                    if round(x7['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g7 = qtd_g7 + 1

        X_gurobi = pd.DataFrame (matrix_final, columns=list_X_est[0].columns.values)
        X_gurobi['UserID'] = indices
        X_gurobi.set_index("UserID", inplace = True)
        
        return X_gurobi