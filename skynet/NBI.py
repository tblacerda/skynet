import datetime
import functools
import pandas as pd
import re
import telnetlib
import time
from collections import defaultdict, deque
from .setup import *
from .huawei import *


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
        self.sites_data = defaultdict(lambda: deque(maxlen=300)) # para registar os dados dos sites
        self.SITES = []
    
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
        
    def _usuarios_site(self):
        """
        Retrieve the total number of users across all cells in the site.

        Parameters:
        connection: The connection object to interact with the network.
        cell: The specific cell to query.

        Returns:
        int: Total number of users across all cells in the site.
        """
        expected_return = "RETCODE = 0  Operation succeeded"
        retcode = self.send_command("DSP CELLUECNT:;")
        if expected_return not in retcode:
            raise Exception(f"Failed to retrieve user count. Error: {retcode}")
        else:
            print(expected_return)
        df = self.parse_CELLUECNT(retcode)
        usuarios = df['Cell Total Counter'].sum()

        return usuarios
    
    def update_usuarios_sites(self):
        '''
        conecta no MAE verifica a quantidade total de usuarios no Site
        '''

        if self.connection is None:
            self.connect()

        for site in self.SITES:
            try:
                self.select_ne(site)
                # consulta a quandidade de usuarios 
                print(f"Atualizando usuarios do site {site}...")
                user_count = self._usuarios_site()
                timestamp = datetime.datetime.now()
                self.sites_data[site].append({'timestamp': timestamp, 'user_count': user_count})
                with open("Sites_users.log", "a") as log_file:
                    log_file.write(f"{timestamp} - Site: {site}, User Count: {user_count}\n")
            except Exception as e:
                print(f"Error updating users for site {site}: {str(e)}")
    
