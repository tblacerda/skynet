import telnetlib
import re
import pandas as pd
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv('settings.ini'))
import schedule
import time
import datetime
import functools
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

TeclaEnter = "\r\n"
LOG_FILE = "telnet_commands.log"
LOG_RESULTS = "results.log"
MAEHOSTNAME = config('MAEHOSTNAME')
MAEPORT = config('MAEPORT')
LOGIN = config('LOGIN')
PASSWORD = config('PASSWORD')

MAXUSERS = {
    "1800_15": 150, 
    "2600_20": 250, # entre 200 e 800
    "2600_10": 50,
    "2100_10": 50,
}
VELOCIDADE = 20 # segundos. Tempo de espera entre comandos
CONFIDENCE = 0.95
TRIALS = 30
TILT_RANGE = (0,15)
AZIMUTH_RANGE = (-30,30)
RS_RANGE_2600 = (45, 95)
RS_RANGE_18002100 = (45, 130)
RS_STEP = 10
REPEAT = 1 # quantas x repete a probe de usuarios e pega o maior valor
SITES  = ['SR-CGLBJ0', 'SR-PB01CW']


# Decorator to log Telnet commands
def log_telnet_commands(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        command = args[1]  # Assuming the first argument after `self` is the command
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Log the command
        log_entry = f"[{timestamp}] Sent command: {command}\n"
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        print(log_entry, end="")  # Also print to console if needed
        # Execute the original function
        return func(*args, **kwargs)
    return wrapper

class NBI:
    """Class to connect via telnet to MAE NBI (host: port 31114)"""

    def __init__(self):
        self.df_global = pd.DataFrame()  # Substitui a variável global
        self.host = MAEHOSTNAME
        self.port = MAEPORT
        self.connection = None
        self.prompt = b"---    END\r\n"
        self.prompt2 = b" reports in total\r\n---    END\r\n"
        self.ne_name = 'None'
        self.ne_vnfc = 'None'
    
    def reset_global_data(self):
        """Reseta os dados acumulados"""
        self.df_global = pd.DataFrame()

    def get_network_status(self) -> None:
        """Get network status and store in instance"""
        try:
            result = self.send_command("DSP CELLUECNT:;")
            df1 = self.parse_CELLUECNT(result)
            result = self.send_command("LST SECTORSPLITCELL:;")
            df2 = self.parse_SECTORSPLITCELL(result)
            df_merged = pd.merge(df1, df2, on='Local Cell ID', how='inner')
            df_merged['Timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Atualiza o DataFrame interno
            self.df_global = df_merged
            # if self.df_global.empty:
            #     self.df_global = df_merged
            # else:
            #     self.df_global = self.df_merged
            
        except Exception as e:
            raise Exception(f"Failed to get network status. Error: {str(e)}")

    def connect(self):
        """Establish telnet connection to the host"""
        try:
            self.connection = telnetlib.Telnet(self.host, self.port, timeout=60)
            result = self.login()
        except Exception as e:
            raise Exception(f"Failed to connect to {self.host}:{self.port}. Error: {str(e)}")
        
    def disconnect(self):
        """Close the telnet connection"""
        try:
            if self.connection:
                self.logout()
                self.connection.close()
                self.connection = None
        except Exception as e:
            raise Exception(f"Failed to disconnect. Error: {str(e)}")
        
    def login(self):
        """Login to the telnet server"""
        #loginstring = 'LGI:OP="F8058552",PWD="Sq+hvz[p(";'
        loginstring = f'LGI:OP="{LOGIN}",PWD="{PASSWORD}";'         
        if self.connection is None:
            raise Exception("Not connected to any server. Please connect first.")
        
        try:
            self.connection.write(loginstring.encode('ascii') + b"\r\n")
            result = self.connection.read_until(self.prompt).decode('ascii')
            
            if "RETCODE = 0  Success" not in result:
                raise Exception(f"Failed to connect to {self.host}:{self.port}. Error: User or Password Incorrect!")
        except Exception as e:
            raise Exception(f"Failed to login. Error: {str(e)}")

    def logout(self):
        """Logout from the telnet server"""
        logoutstring = 'LGO:OP="SCORE001";'

        if self.connection is None:
            raise Exception("Not connected to any server. Please connect first.")
        
        try:
            self.connection.write(logoutstring.encode('ascii') + b"\r\n")
        except Exception as e:
            raise Exception(f"Failed to logout. Error: {str(e)}")

    def reg_vnfc(self, vnfc_name):
        """Register VNFC with given name"""
        regvnfcstring = f'REG VNFC:NAME="{vnfc_name}";'
        result = self.send_command(regvnfcstring)
        
        if "RETCODE = 0  Success" not in result:
            raise Exception(f"Failed to connect to {vnfc_name}. Error: VNFC Invalid Name")

    def _reg_ne(self, ne_name):
        """Register NE with given name (internal use only)"""
        self.ne_name = ne_name
        regnestring = f'REG NE:NAME={ne_name};'
        result = self.send_command(regnestring)
        
        if "RETCODE = 0  Success" not in result:
            raise Exception(f"Failed to connect to {ne_name}. Error: {result}")

    def _unreg_ne(self, ne_name):
        """Unregister NE with given name (internal use only)"""
        regnestring = f'UNREG NE:NAME={ne_name};'
        result = self.send_command(regnestring)
        
        if "RETCODE = 0  Success" not in result:
            raise Exception(f"Failed to unregister {ne_name}. {result}")

    def select_ne(self, ne_name):
        """Select NE, ensuring only one NE is selected at a time"""
        if self.ne_name != 'None':
            self._unreg_ne(self.ne_name)
        self._reg_ne(ne_name)

    @log_telnet_commands
    def send_command(self, command):
        """Send command to the telnet server and return response"""
        if self.connection is None:
            raise Exception("Not connected to any server. Please connect first.")

        try:
            #print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Command: {command}")   
            
            self.connection.write(command.encode('ascii') + b"\r\n")
            response = self.connection.read_until(self.prompt).decode('ascii')
            #self.connection.write(TeclaEnter.encode('ascii') + b"\r\n") #tiago
            if "To be continued..." in response:
                response = response + self.connection.read_until(self.prompt2).decode('ascii')
            elif "RETCODE = 20111  There is no data in the table" in response:
                raise Exception(f"Empty Response: {self.ne_name} [{command}]. Error: RETCODE = 20111  There is no data in the table")
            elif "RETCODE = 1  Failed to query NE information" in response:
                raise Exception(f"Failed To Query: {self.ne_name} [{command}]. Error: RETCODE = 1  Failed to query NE information")

            return response.strip()
        except Exception as e:
            raise Exception(f"Failed to send command: {self.ne_name} [{command}]. Error: {str(e)}")

    @log_telnet_commands
    def send_command_confirm(self, command):
    # print(f"Send {command}")
        if self.connection is None:
            raise Exception("Not connected to any server. Please connect first.")
        try:
            confirm_prompt = "Y/N" ####### String de confirmação
            # Write the command
            self.connection.write((command).encode('ascii') + b"\r\n")
            # Read the response
            time.sleep(10)
            response = self.connection.read_lazy()
            # print(f"Initial response: {response}")  # Debug print

            # response = self.connection.read_until(confirm_prompt.encode('ascii')).decode('ascii')

            print(f"Initial response: {response}")  # Debug print
            self.connection.write(b"Y\r\n")
            response += self.connection.read_until(self.prompt).decode('ascii')
            print(f"Final response: {response}")  # Debug print
            if "To be continued..." in response:
                response = response + self.connection.read_until(self.prompt2).decode('ascii')
            elif "RETCODE = 20111  There is no data in the table" in response :
                raise Exception(f"Empty Response: {self.ne_name} [{command}]. Error: RETCODE = 20111  There is no data in the table")
            elif "RETCODE = 1  Failed to query NE information" in response:
                raise Exception(f"Failed To Query: {self.ne_name} [{command}]. Error: RETCODE = 1  Failed to query NE information")
            # print(response.strip())
            return response.strip()
        except Exception as e:
            # print("levantei erro de comando")
            raise Exception(f"Failed to send command: {self.ne_name} [{command}]. Error: {str(e)}")

    def get_ne_list(self):
        """Get list of Network Elements"""

        def parse_ne_list(result):
            """Parse NE list result into a dictionary"""
            ne_list = {}
            lines = result.splitlines()
            for line in lines:
                if line.startswith("NE Type"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    ne_type = parts[0]
                    ne_name = parts[1]
                    ip_address = parts[2]
                    ne_list[ne_name] = {"type": ne_type, "ip": ip_address}
            return ne_list

        try:
            self.connect()
            result = self.send_command("LST NE:;")
            #dp = DataParser()
            result = parse_ne_list(result)
        except Exception as e:
            raise Exception(f"Failed to get NE list. Error: {str(e)}")
        return result

    def parse_CELLUECNT(self, result):
        # Regular expression to match the data rows
        pattern = re.compile(r'(\d+)\s+([\w-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
        matches = pattern.findall(result)

        # Create a DataFrame directly from the matches
        df = pd.DataFrame(matches, columns=[
            'Local Cell ID',
            'Cell Name',
            'Cell MO-Signal Counter',
            'Cell MO-Data Counter',
            'Cell MT-Access Counter',
            'Cell Other Counter',
            'Cell Total Counter',
            'Cell VoLTE Counter',
            'Cell CA Counter'
        ])

        # Convert columns to appropriate data types (optional)
        df = df.astype({
            'Local Cell ID': int,
            'Cell MO-Signal Counter': int,
            'Cell MO-Data Counter': int,
            'Cell MT-Access Counter': int,
            'Cell Other Counter': int,
            'Cell Total Counter': int,
            'Cell VoLTE Counter': int,
            'Cell CA Counter': int
        })

        return df
    
    def parse_SECTORSPLITCELL(self, log_text):
        # Encontrar a tabela usando regex
        table_match = re.search(r"List Sector Splitting Cell.*?(-+)(.*?)\(Number of results", log_text, re.DOTALL)
        if not table_match:
            raise ValueError("Tabela não encontrada no log.")
        
        table_content = table_match.group(2).strip()
        
        # Separar cabeçalhos e linhas de dados
        lines = table_content.split('\n')
        headers = re.split(r'\s{2,}', lines[0].strip())
        data_lines = lines[2:]  # Ignorando linha de cabeçalho e separador
        
        # Processar linhas de dados
        data = []
        for line in data_lines:
            values = re.split(r'\s{2,}', line.strip())
            data.append(values)
        
        # Criar DataFrame
        df = pd.DataFrame(data, columns=headers)
        df.rename(columns={'Local cell ID': 'Local Cell ID'}, inplace=True)
        # Enforce columns as int32
        int_columns = [col for col in df.columns if col != 'Cell Beamwidth']
        for col in int_columns:
            df[col] = df[col].astype('int32')
        return df
        
def is_MIMO(cell_name):
    """Check the type of the cell based on its name."""
    if is_2600_20MHz(cell_name):
        return True
    elif is_2600_10MHz(cell_name):
        return True
    elif is_1800_15MHz(cell_name):
        return True
    elif is_2100_10MHz(cell_name):
        return True
    else:
        return False

def get_setor_number(cell_name):
    """Return the sector number: 1 for A/I, 2 for B/J, 3 for C/K."""

    def is_setor_A(cell_name):
        """Check if the cell name ends with sector A or sector I."""
        return any(cell_name.endswith(ending) for ending in ["A", "I"])

    def is_setor_B(cell_name):
        """Check if the cell name ends with sector B or sector J."""
        return any(cell_name.endswith(ending) for ending in ["B", "J"])

    def is_setor_C(cell_name):
        """Check if the cell name ends with sector C or sector K."""
        return any(cell_name.endswith(ending) for ending in ["C", "K"])

    if is_setor_A(cell_name):
        return 1
    elif is_setor_B(cell_name):
        return 2
    elif is_setor_C(cell_name):
        return 3
    else:
        return None  # Return None if no sector matches

def beam_number(cell_name):
    """Return the beam number for the given cell.
    for 1A, 1B, 1C, 1I, 1J, 1K, return 1
    for 2A, 2B, 2C, 2I, 2J, 2K, return 2
    for 3A, 3B, 3C, 3I, 3J, 3K, return 3
    for 4A, 4B, 4C, 4I, 4J, 4K, return 4
    """
    return int(cell_name[-2])

def primary_cell(cell_name):
    """
    return TRUE if it is a primary cell
    """
    return is_1800_15MHz(cell_name) or is_2600_20MHz(cell_name)

def secondary_cell(cell_name):
    """
    Return the cell name for secondary cell for a given primary cell.
    Returns None if the cell given is not a primary cell.
    
    Primary cells have formats like:
    - '4G-RCRCJ0-26-[1-4]A/B/C' - secondary cells are '4G-RCRCJ0-26-[1-4]I/J/K'
    - '4G-RCRCJ0-18-[1-4]A/B/C' - secondary cells are '4G-RCRCJ0-21-[1-4]A/B/C'
    """
    # For 2600 MHz cells
    if "-26-" in cell_name:
        # Check the last character for sector letter
        if cell_name.endswith("A"):
            return cell_name[:-1] + "I"
        elif cell_name.endswith("B"):
            return cell_name[:-1] + "J"
        elif cell_name.endswith("C"):
            return cell_name[:-1] + "K"
    
    # For 1800 MHz cells
    elif "-18-" in cell_name:
        if cell_name.endswith(("A", "B", "C")):
            # Replace 18 with 21 but keep the same sector letter
            return cell_name.replace("-18-", "-21-")
    
    # Not a primary cell or unrecognized format
    return None

def is_2600_20MHz(cell_name):
    """Check if the cell name ends with any of the specified strings."""
    special_endings = ["26-1A",
                       "26-1B",
                       "26-1C",
                       "26-2A",
                       "26-2B",
                       "26-2C",
                       "26-3A",
                       "26-3B",
                       "26-3C", 
                       "26-4A",
                       "26-4B",
                       "26-4C",
                       "26-1D",
                       "26-2D",
                       "26-3D",
                       "26-4D"]
    return any(cell_name.endswith(ending) for ending in special_endings)

def is_2600_10MHz(cell_name):
    """Check if the cell name ends with any of the specified strings."""
    special_endings = ["26-1I",
                       "26-1J",
                       "26-1K",
                       "26-2I",
                       "26-2J",
                       "26-2K",
                       "26-3I",
                       "26-3J",
                       "26-3K", 
                       "26-4I",
                       "26-4J",
                       "26-4K"]
    return any(cell_name.endswith(ending) for ending in special_endings)

def is_1800_15MHz(cell_name):
    """Check if the cell name ends with any of the specified strings."""
    special_endings = ["18-1A",
                       "18-1B",
                       "18-1C",
                       "18-2A",
                       "18-2B",
                       "18-2C",
                       "18-3A",
                       "18-3B",
                       "18-3C", 
                       "18-4A",
                       "18-4B",
                       "18-4C",
                       "18-1D",
                       "18-2D",
                       "18-3D",
                       "18-4D"]
    return any(cell_name.endswith(ending) for ending in special_endings)

def is_2100_10MHz(cell_name):
    """Check if the cell name ends with any of the specified strings."""
    special_endings = ["21-1A",
                       "21-1B",
                       "21-1C",
                       "21-2A",
                       "21-2B",
                       "21-2C",
                       "21-3A",
                       "21-3B",
                       "21-3C", 
                       "21-4A",
                       "21-4B",
                       "21-4C",
                       "21-1D",
                       "21-2D",
                       "21-3D",
                       "21-4D"]
    return any(cell_name.endswith(ending) for ending in special_endings)

def MaxUsers(cell_name):
    """Return the maximum number of users for the given cell."""
    if is_2600_20MHz(cell_name):
        return MAXUSERS["2600_20"]
    elif is_2600_10MHz(cell_name):
        return MAXUSERS["2600_10"]
    elif is_1800_15MHz(cell_name):
        return MAXUSERS["1800_15"]
    elif is_2100_10MHz(cell_name):
        return MAXUSERS["2100_10"]
    else:
        return 250 

def latin_hypercube(n_samples, dim, bounds):
    """
    Gera amostras usando Latin Hypercube Sampling.
    Dentro dos intervalos: np.random.uniform escolhe um ponto aleatório em cada subintervalo.

    Ordem dos intervalos: np.random.shuffle embaralha a ordem dos intervalos.

    Resultado: Pontos distribuídos uniformemente, mas com variação entre execuções (a menos que você fixe a seed).
    Parâmetros:
        n_samples (int): Número de amostras.
        dim (int): Dimensão do espaço (2 para tilt/azimute).
        bounds (list): Lista de tuplas com limites [(min, max), ...].
    
    Retorna:
        np.array: Amostras no espaço normalizado [0, 1]^dim.

    """
    samples = np.zeros((n_samples, dim))
    for i in range(dim):
        # Divide o espaço em n_samples intervalos
        intervals = np.linspace(0, 1, n_samples + 1)
        # Amostra aleatoriamente dentro de cada intervalo
        points = np.random.uniform(intervals[:-1], intervals[1:], size=n_samples)
        # Embaralha para garantir aleatoriedade
        np.random.shuffle(points)
        samples[:, i] = points
    return samples

def maximize_users(connection,
                    user_function,
                    cell, 
                    local_cell_id,
                    sector_split_group_id,
                    tilt_atual,
                    azimute_atual,
                    n_iter = TRIALS, init_points=2, noise_level=1e-2,
                    stop_confidence = CONFIDENCE 
                    ): #1e-6
    """Bayesian optimization with Gaussian Process and Expected Improvement."""
    t_bounds = (TILT_RANGE[0], TILT_RANGE[1]) #-2 para geral e -15 para estádios
    a_bounds = (AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])
    bounds = np.array([t_bounds, a_bounds])
    
    # Initial points based on predetermined positions
#    initial_positions = [(2, -15), (2, 15), (10, 0) ]
    initial_positions = [(tilt_atual, azimute_atual),(6,0)]
    
    X = np.array(initial_positions[:init_points])
    y = np.array([user_function(connection, cell, t, a, local_cell_id, sector_split_group_id) for t, a in X])
    evaluated_points = set(map(tuple, X))
    
    # Kernel with optimized hyperparameters and noise level (alpha)
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_level,  # Initial noise level
        n_restarts_optimizer=50,  # Optimize kernel hyperparameters and alpha
        normalize_y=True
    )
    gp.fit(X, y)
    
    def expected_improvement(x):
        """Vectorized Expected Improvement."""
        x = np.atleast_2d(x)
        mu, sigma = gp.predict(x, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        y_max = np.max(y)
        improvement = mu - y_max
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei  # Negative for minimization
    
    # Bayesian optimization loop
    for iteracoes in range(n_iter): # até 30 chamadas a função de usuário (acesso ao HUAWEI e aguarda x Segundos)
        # Find candidate with the highest EI using gradient-based optimization
        best_x = None
        best_ei = np.inf  # We minimize -EI
        # Geração de pontos iniciais via Latin Hypercube
        lhs_samples = latin_hypercube(n_samples=250, dim=2, bounds=[t_bounds, a_bounds])
        
        # Normalização para os limites reais
        t_scaled = lhs_samples[:, 0] * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        a_scaled = lhs_samples[:, 1] * (a_bounds[1] - a_bounds[0]) + a_bounds[0]
        
        # Conversão para inteiros (mantendo precisão para otimização contínua)
        t_ints = np.round(t_scaled).astype(int)
        a_ints = np.round(a_scaled).astype(int)
        
        # Garantia de limites físicos
        t_ints = np.clip(t_ints, t_bounds[0], t_bounds[1])
        a_ints = np.clip(a_ints, a_bounds[0], a_bounds[1])

        # Multi-start optimization to avoid local minima
        # calcula-se o valor da funçao estimada em todos os 50 pontos e escolhe-se o melhor
        # o melhor é o que tiver maior Expected Improvement
        # o maior EI é o que tem maior incerteza, que é o que está na borda do intervalo de confianca frequentista
        # vide figura 1 de A Tutorial on Bayesian Optimization.
        for i in range(250):  # 5
            x0 = np.array([t_ints[i], a_ints[i]], dtype=np.float64)
            res = minimize(expected_improvement, x0, bounds=bounds, method='L-BFGS-B')
            
            if res.fun < best_ei and not any(np.allclose(res.x, x, atol=1e-3) for x in evaluated_points):
                best_ei = res.fun
                best_x = res.x
        
        if best_x is None:
            continue  # Fallback to random sampling if all candidates are duplicates
        
            # Early stopping check: Probability of Improvement
        mu, sigma = gp.predict(best_x.reshape(1, -1), return_std=True)
        y_max = np.max(y)
        z = (y_max - mu) / sigma
        prob_improvement = 1 - norm.cdf(z)
        if prob_improvement < (1 - stop_confidence):
            break  # Para se o incremento esperado for menor que 5%
    
        # Arredondamento final para posições físicas
        best_x = np.round(best_x, decimals=1)
        # Evaluate and update

        new_y = user_function(connection, cell, best_x[0], round(best_x[1]), local_cell_id, sector_split_group_id)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Cell: {cell}; Tilt: {best_x[0]}; Azimuth: {round(best_x[1])};  Usuarios: {new_y}; Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement)}\n"
        #print(f"[{timestamp}] Cell: {cell}; Tilt: {(best_x[0])}; Azimuth: {(best_x[1])};  Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement)}")
        with open(LOG_RESULTS, "a") as f:
             f.write(log_entry)
        X = np.vstack([X, best_x])
        y = np.append(y, new_y)
        evaluated_points.add(tuple(best_x))
        gp.fit(X, y)  # Re-fit GP with updated data and optimized alpha
    
    idx_best = np.argmax(y)
    # Before logging
    prob_improvement = float(prob_improvement)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]  Melhor posicao ; Cell: {cell}; Tilt: {(X[idx_best, 0])}; Azimuth: {int(X[idx_best, 1])};  Iterations: {iteracoes + 1}; Confidence: {(1 - prob_improvement):.4f}\n"
    with open(LOG_RESULTS, "a") as f:
        f.write(log_entry)
    return ((X[idx_best, 0]), round(X[idx_best, 1])), y[idx_best]

# Função hipotética que retorna o número de usuários para uma posição
def FunctionUsers(connection, cell, tilt, azimuth, local_cell_id, sector_split_group_id, repeat= REPEAT):
    # Simula o cálculo do número de usuários para uma dada posição
    max_usuarios = 0  # Initialize maximum users count
    signal = 1 # if tilt >= 0 else -1
    tilt_int = int(tilt)  # get integer part of tilt
    tilt_frac = signal * round((tilt - tilt_int) * 10)  # get fractional part (tenths)

    ExpectedReturn = "RETCODE = 0  Operation succeeded"
    RETCODE = connection.send_command(f"MOD SECTORSPLITCELL:LOCALCELLID={local_cell_id},SECTORSPLITGROUPID={sector_split_group_id},CELLBEAMTILT={tilt_int},CELLBEAMTILTFRACTIONPART={tilt_frac},CELLBEAMAZIMUTHOFFSET={azimuth};")
    if ExpectedReturn not in RETCODE:
        raise Exception(f"Failed to modify sector split cell. Error: {RETCODE}")
    else:
        print(ExpectedReturn)
    for _ in range(repeat):
        time.sleep(VELOCIDADE)
        #RETCODE = connection.send_command(TeclaEnter)
        RETCODE = connection.send_command("DSP CELLUECNT:;")
        if ExpectedReturn not in RETCODE:
            raise Exception(f"Failed to modify sector split cell. Error: {RETCODE}")
        else:
            print(ExpectedReturn)    
        df = connection.parse_CELLUECNT(RETCODE)
        Usuarios = df.loc[df['Cell Name'] == cell, 'Cell Total Counter'].values[0]
        max_usuarios = max(max_usuarios, Usuarios)  # Keep track of maximum value
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Cell {cell} has {Usuarios} users at tilt {tilt} and azimuth {azimuth}")

    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{timestamp}] Cell {cell} has {max_usuarios} users at tilt {tilt} and azimuth {azimuth}")

    return max_usuarios

def AjusteDePotencia(sentido, localcellid, cell):
    """
    Ajusta a potência de transmissão para os usuários em uma célula específica.

    Parâmetros:
    usuarios (list): Lista de usuários que precisam de ajuste de potência.
    localcellid (int): Identificador da célula local onde o ajuste de potência será aplicado.

    Retorna:
    None
    """
    def ParserRsSignal(text):
    
        # Split the input string into lines
        lines = text.split('\r\n')

        # Initialize the value as None
        reference_signal_power = None

        # Iterate through each line to find the target parameter
        for line in lines:
            if 'Reference signal power(0.1dBm)' in line:
                # Split the line into key and value parts
                parts = line.split('=')
                if len(parts) >= 2:
                    # Extract and clean the value
                    reference_signal_power = int(parts[1].strip())
                break  # Exit loop once found

        #print(f"Reference signal power(0.1dBm): {reference_signal_power}")
        return int(reference_signal_power)

    if is_1800_15MHz(cell) or is_2100_10MHz(cell):
        RS_RANGE = RS_RANGE_18002100
    else:
        RS_RANGE = RS_RANGE_2600

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if sentido == 'reduzir':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        RsAtual = ParserRsSignal(Mae.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RsAtual - RS_STEP >= RS_RANGE[0]:
            RsNovo = RsAtual - RS_STEP
            Mae.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
        else:
            log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal já no mínimo = {RS_RANGE[0]} dBm\n"

    elif sentido == 'aumentar':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        RsAtual = ParserRsSignal(Mae.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RsAtual + RS_STEP <= RS_RANGE[1]:
            RsNovo = RsAtual + int(RS_STEP/2)
            Mae.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
        else:
            log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal ja no maximo = {RS_RANGE[1]} dBm\n"
    else:
        log_entry = f"[{timestamp}] ERRO NO AJUSTE DE RS\n"

    print(log_entry)
    with open(LOG_RESULTS, "a") as f:
        f.write(log_entry)

def AjustForBestPos(connection,
                    cell,
                    tilt,
                    azimuth,
                    best_val,
                    local_cell_id,
                    sector_split_group_id ) -> None:

    signal = 1 if tilt >= 0 else -1
    tilt_int = int(tilt)  # get integer part of tilt
    tilt_frac = signal * round((tilt - tilt_int) * 10)  # get fractional part (tenths)

    connection.send_command(f"MOD SECTORSPLITCELL:LOCALCELLID={local_cell_id},SECTORSPLITGROUPID={sector_split_group_id},CELLBEAMTILT={tilt_int},CELLBEAMTILTFRACTIONPART={tilt_frac},CELLBEAMAZIMUTHOFFSET={azimuth};")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Melhor posição {cell}: TILT={tilt}, AZIMUTE={azimuth} com {best_val} usuários\n"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

if __name__ == "__main__":
    
    df_report = pd.DataFrame()
    Mae = NBI()
    Mae.connect()

    while True:
        now = datetime.datetime.now()
        # Define the allowed interval: between 17:00 and 05:00 (overnight)
        if now.hour >= 13 or now.hour < 23:
            for site in SITES:
                Mae.select_ne(site)
                Mae.get_network_status()
                cells = set(Mae.df_global['Cell Name'])
                if site == 'SR-CGLBJ0':
                    cells = [
                        "4C-CGLBJ0-26-1D",
                        "4C-CGLBJ0-26-2D",
                        "4C-CGLBJ0-26-3D",
                        "4C-CGLBJ0-26-4D",
                        "4C-CGLBJ0-26-1W",
                        "4C-CGLBJ0-26-2W",
                        "4C-CGLBJ0-26-3W",
                        "4C-CGLBJ0-26-4W",
                        "4G-CGLBJ0-18-1D",
                        "4G-CGLBJ0-18-2D",
                        "4G-CGLBJ0-18-3D",
                        "4G-CGLBJ0-18-4D",
                        "4G-CGLBJ0-21-1D",
                        "4G-CGLBJ0-21-2D",
                        "4G-CGLBJ0-21-3D",
                        "4G-CGLBJ0-21-4D"
                    ]

                try:
                    for cell in (cells):
                        if primary_cell(cell):
                            #### FIM CONTROLE DE POTENCIA ####        
                            for cell1 in (cells):
                                local_cell_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell1, 'Local Cell ID'].values[0]
                                usuarios = Mae.df_global[Mae.df_global['Cell Name'] == cell1]['Cell Total Counter'].values[0]
                                if usuarios < 0.8 * MaxUsers(cell1):
                                    AjusteDePotencia('aumentar', local_cell_id, cell1)
                                elif usuarios > 1.1 * MaxUsers(cell1):
                                    AjusteDePotencia('reduzir', local_cell_id, cell1)
                            #### FIM CONTROLE DE POTENCIA ####
                            local_cell_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell, 'Local Cell ID'].values[0]
                            sector_split_group_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell, 'Sector Split Group ID'].values[0]                        
                            tilt_atual = Mae.df_global[Mae.df_global['Cell Name'] == cell]['Cell Beam Tilt(degree)'].values[0]
                            azimute_atual = Mae.df_global[Mae.df_global['Cell Name'] == cell]['Cell Beam Azimuth Offset(degree)'].values[0]
                            usuarios = Mae.df_global[Mae.df_global['Cell Name'] == cell]['Cell Total Counter'].values[0]
                            if usuarios < MaxUsers(cell):
                                AjusteDePotencia('aumentar', local_cell_id, cell)
                                best_pos, best_val = maximize_users(Mae,
                                                                    FunctionUsers,
                                                                    cell,
                                                                    local_cell_id,
                                                                    sector_split_group_id,
                                                                    tilt_atual,
                                                                    azimute_atual
                                                                    )
                                
                                AjustForBestPos(Mae,
                                                cell,
                                                best_pos[0],
                                                best_pos[1],
                                                best_val,
                                                local_cell_id,
                                                sector_split_group_id)
                                
                                try:  
                                    SecondaryCell = secondary_cell(cell)
                                    usuarios = Mae.df_global[Mae.df_global['Cell Name'] == SecondaryCell]['Cell Total Counter'].values[0]
                                    local_cell_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == SecondaryCell, 'Local Cell ID'].values[0]
                                    if usuarios < MaxUsers(SecondaryCell):
                                        AjusteDePotencia('aumentar', local_cell_id, cell)
                                    elif usuarios > MaxUsers(SecondaryCell):
                                        AjusteDePotencia('reduzir', local_cell_id, cell)
                                    else:
                                        continue

                                    try:
                                        local_cell_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == SecondaryCell, 'Local Cell ID'].values[0]
                                        sector_split_group_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == SecondaryCell, 'Sector Split Group ID'].values[0]

                                    except:
                                        if SecondaryCell.startswith('4C-'):
                                            SecondaryCell = '4G-' + SecondaryCell[3:]
                                            local_cell_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == SecondaryCell, 'Local Cell ID'].values[0]
                                            sector_split_group_id = Mae.df_global.loc[Mae.df_global['Cell Name'] == SecondaryCell, 'Sector Split Group ID'].values[0]
                                        else:
                                            pass
                                        
                                    AjustForBestPos(Mae,
                                                SecondaryCell,
                                                best_pos[0],
                                                best_pos[1],
                                                best_val,
                                                local_cell_id,
                                                sector_split_group_id)
                                except Exception as e:
                                    print(f"Erro: {str(e)}")
                                    continue
                                
                            else:
                                AjusteDePotencia('reduzir', local_cell_id, cell)
                                        
                        else:
                            continue
                        Mae.get_network_status()
                        df_report = pd.concat([df_report, Mae.df_global], ignore_index=True)
                        df_report.to_excel(r'Report.xlsx', index=False)
                except Exception as e:
                    print(f"Erro: {str(e)}")
                    continue
                Mae.reset_global_data()        
            break  # Exit the while loop after running the FOR loop
        else:
            print("Current time is outside the allowed interval (17:00-05:00). Waiting 15 minutes...")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_RESULTS, "a") as f:
                f.write(f"[{timestamp}] Current time is outside the allowed interval (17:00-05:00). Waiting 15 minutes...\n")
            time.sleep(15 * 60)

    Mae.disconnect()


