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
    "700_10": 50,
}

RS_MIN = 40
RS_STEP = 5
RefPwr = pd.read_excel(r'CELULAS_PWR_reduzido.xlsx')
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
            users = self.send_command("DSP CELLUECNT:;")
            dfUsers = self.parse_CELLUECNT(users)
            RSAtual = self.send_command("LST PDSCHCFG:;")
            dfRS = parse_pdsch_data(RSAtual)
            df = pd.merge(dfUsers, dfRS, on='Local Cell ID', how='inner')
            # Atualiza o DataFrame interno
            self.df_global = df
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
                       "26-4D",
                       "26-A",
                       "26-B",
                       "26-C",]
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
                       "26-4K",
                       "26-I",
                       "26-J",
                       "26-K",]
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
                       "18-4D",
                       "18-A",
                       "18-B",
                       "18-C"]
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
                       "21-4D",
                       "21-A",
                       "21-B",
                       "21-C",]
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
        return 50

def AjusteDePotencia(sentido, localcellid, cell, RS_MAX, RSAtual):
    """
    Ajusta a potência de transmissão para os usuários em uma célula específica.

    Parâmetros:
    usuarios (list): Lista de usuários que precisam de ajuste de potência.
    localcellid (int): Identificador da célula local onde o ajuste de potência será aplicado.

    Retorna:
    None
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = ""
    if sentido == 'reduzir':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        #RsAtual = ParserRsSignal(Mae.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RSAtual - RS_STEP >= RS_MIN:
            RsNovo = RSAtual - RS_STEP
            Mae.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
            with open(LOG_RESULTS, "a") as f:
                f.write(log_entry)
        else:
            pass
            #log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal ja no mínimo = {RS_MIN} dBm\n"

    elif sentido == 'aumentar':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        #RsAtual = ParserRsSignal(Mae.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RSAtual + RS_STEP <= RS_MAX:
            RsNovo = RSAtual + int(RS_STEP)
            Mae.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
            with open(LOG_RESULTS, "a") as f:
                f.write(log_entry)
        else:
            pass
            #log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal ja no maximo = {RS_MAX} dBm\n"
    else:
        log_entry = f"[{timestamp}] ERRO NO AJUSTE DE RS\n"
        with open(LOG_RESULTS, "a") as f:
            f.write(log_entry)
    
    print(log_entry)
 
def parse_pdsch_data(data_string):
    # Verificar se existem blocos de dados
    if "Local cell ID" not in data_string:
        return pd.DataFrame()

    # Definir cabeçalhos fixos (conforme estrutura conhecida)
    headers = [
        'Local Cell ID',
        'Reference signal power(0.1dBm)',
        'PB',
        'Reference Signal Power Margin(0.1dB)',
        'Offset of Ant0 to Tx Power(0.1dB)',
        'Offset of Ant1 to Tx Power(0.1dB)',
        'Offset of Ant2 to Tx Power(0.1dB)',
        'Offset of Ant3 to Tx Power(0.1dB)',
        'TX Channel Power Config Switch',
        'Cell Power Limit(0.01W)',
        'EMF Power Limit Switch',
        'CRS Power Boosting Amplitude',
        'Logical Port Swap Switch',
        'PDSCH Power Boosting Switch',
        'CRS Power Reduction Amount'
    ]
    
    # Encontrar todos os blocos de dados (considerando múltiplas ocorrências)
    blocks = re.findall(
        r'Local cell ID[\s\S]*?(?=\r?\n\(Number of results = \d+\)|$)', 
        data_string
    )
    
    all_data = []
    
    for block in blocks:
        # Extrair apenas linhas de dados (que começam com número)
        data_lines = re.findall(r'^\d+\s+.*$', block, re.MULTILINE)
        
        for line in data_lines:
            # Dividir usando múltiplos espaços como separador
            parts = re.split(r'\s{2,}', line.strip())
            
            # Validar número mínimo de colunas
            if len(parts) < len(headers):
                continue
                
            # Processar campos especiais
            processed = []
            for i, part in enumerate(parts):
                # Campos booleanos (On/Off)
                if i in [10, 12, 13]:
                    processed.append(1 if part.strip().lower() == 'on' else 0)
                
                # Campos com unidades dB
                elif i in [11, 14]:
                    value = part.replace('dB', '').strip()
                    processed.append(float(value) if value.replace('.', '', 1).isdigit() else 0.0)
                
                # Campo TX Channel Power (preservar string)
                elif i == 8:
                    processed.append(part)
                
                # Campos numéricos
                else:
                    if part == 'Off':
                        processed.append(0)
                    else:
                        try:
                            processed.append(int(part))
                        except ValueError:
                            processed.append(part)
            
            # Garantir que temos exatamente 15 campos
            if len(processed) == len(headers):
                all_data.append(processed)
    
    # Convert the list of lists to a list of dictionaries
    
    # Keep only the required columns
    filtered_data = []
    for row in all_data:
        data_dict = dict(zip(headers, row))
        filtered_data.append({
            'Local Cell ID': data_dict['Local Cell ID'],
            'Reference signal power(0.1dBm)': data_dict['Reference signal power(0.1dBm)']
        })
    return pd.DataFrame(filtered_data)


if __name__ == "__main__":
    
    df_report = pd.DataFrame()
    # Create a dictionary mapping CELLNAME to RSPOWER
    cellname_rspower_dict = dict(zip(RefPwr['CELLNAME'], RefPwr['RSPOWER']))
    sites = set(RefPwr['SITE'].tolist())
    cells = RefPwr['CELLNAME'].tolist()
    Mae = NBI()
    Mae.connect()
    while True:
        for site in set(sites):
            #time.sleep(1)
            # Filter cells belonging to the current site
            try:
                Mae.select_ne(site)
                site_cells = RefPwr[RefPwr['SITE'] == site]['CELLNAME'].tolist()
                Mae.get_network_status()
                
                for cell in site_cells:                
                    RS_MAX = cellname_rspower_dict.get(cell, 180)
                    RS_MIN = int(RS_MAX /3)
                    localcellid = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell, 'Local Cell ID'].values[0]
                    if localcellid is not None:
                        # Check the number of users in the cell
                        num_users = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell, 'Cell Total Counter'].values[0]
                        RSAtual = Mae.df_global.loc[Mae.df_global['Cell Name'] == cell, 'Reference signal power(0.1dBm)'].values[0]
                        if num_users > 1.1 * MaxUsers(cell):
                            AjusteDePotencia('reduzir', localcellid, cell, RS_MAX, RSAtual)
                        elif num_users < 0.9 * MaxUsers(cell):
                            AjusteDePotencia('aumentar', localcellid, cell, RS_MAX, RSAtual)
                        else:
                            pass
                
            except Exception as e:
                print("Error: ", site)
                Mae = NBI()
                Mae.connect()
  