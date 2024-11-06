import hashlib

class FlowerClient(fl.client.NumPyClient):
    """Classe do cliente para o aprendizado federado."""
    def __init__(self, cid, net, trainloader, valloader, group):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.group = group

    def hash_group_id(self, group):
        """Gera um hash do identificador de grupo com sal fixo."""
        salt = b'secure_salt_value'  # Sal fixo compartilhado com o servidor
        group_str = f"{group}".encode()
        hash_object = hashlib.pbkdf2_hmac('sha256', group_str, salt, 100000)
        return hash_object.hex()

    def fit(self, parameters, config):
        """Aplica o treinamento local em um cliente."""
        # Processamento do treinamento local
        hashed_group = self.hash_group_id(self.group)
        metrics = {"group_id": hashed_group, "loss": 0.01}  # Loss é exemplo
        # Retorna parâmetros e métricas...
        return get_parameters(self.net), len(self.trainloader), metrics

# ===================

class FedCustom(fl.server.strategy.Strategy):
    """Estratégia personalizada para agregação de modelos."""
    def __init__(self):
        super().__init__()
        self.hash_to_group_map = {
            # hash(hexstring): original group identifier (int)
        }

    def rebuild_group_map(self):
        """Reconstroi a tabela de lookup segura."""
        salt = b'secure_salt_value'
        original_groups = [1, 2, 3, 4, 5]  # Exemplos de IDs de grupos reais
        self.hash_to_group_map = {
            hashlib.pbkdf2_hmac('sha256', f"{grp}".encode(), salt, 100000).hex(): grp for grp in original_groups
        }

    def aggregate_fit(self, server_round, results, failures):
        """Agrega parâmetros de modelos treinados."""
        if not self.hash_to_group_map:
            self.rebuild_group_map()

        G_AGE = {}

        for i, (client, fit_res) in enumerate(results):
            hashed_group_id = fit_res.metrics.get("group_id")
            group_id = self.hash_to_group_map.get(hashed_group_id)

            if group_id is not None:
                if group_id not in G_AGE:
                    G_AGE[group_id] = []
                G_AGE[group_id].append(i)

        # Continuar restante do processo de agregação...

        print(f"Grupos mapeados: {G_AGE}")



# Hash com Sal Fixo: Ambos os lados (cliente e servidor) devem conhecer e usar o mesmo sal fixo durante a operação do hash, permitindo reproduzir consistentemente o hash previsto e mapeá-lo corretamente aos grupos originais.

# Tabela de Look-up: rebuild_group_map constrói um dicionário hash válido para uso no servidor para entender onde cada hash aponta em termos de identificador real do grupo.

# Risco de Segurança: Note que o uso de sal fixo expõe uma vulnerabilidade caso o sal e o hash sejam comprometidos. Isso deve ser manejado com uma linha de defesa autorizando apenas servidores de confiança e mantendo essa informação fora dos ataques superficiais.
