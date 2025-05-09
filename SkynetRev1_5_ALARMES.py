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
from typing import List, Dict

import decouple
decouple.__version__

TeclaEnter = "\r\n"
LOG_FILE = "telnet_commands.log"
LOG_RESULTS = "results.log"
MAEHOSTNAME = config('MAEHOSTNAME')
MAEPORT = config('MAEPORT')
LOGIN = config('LOGIN')
PASSWORD = config('PASSWORD')
ZAPNUMBER = config('ZAPNUMBER')

SITES = [ 'SR-CE05CW']


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


    def get_alarms(self) -> pd.DataFrame:
        """
        Recebe uma string contendo um ou mais blocos de ALARM no formato do exemplo
        e retorna um DataFrame com uma linha por alarme e colunas para cada campo.

        Campos extraídos:
        • alarm_id, alarm_class, severity, domain, code, category   (da linha “ALARM …”)
        • sync_serial_no, alarm_name, alarm_raised_time,
            alarm_notify_time, location_info, function, special_info,
            special_info1, root_cause_flag, clt, lck                  (das linhas “     Key = Value”)
        • report_device, report_time                                (da linha “+++ DEVICE YYYY-MM-DD hh:mm:ss” acima de cada
            grupo de alarms)
        """

        # 1) Primeiro, separamos os blocos de relatório (cada “+++ … END”)
        #    para poder herdar o device e timestamp de cada relatório.
        
        resposta = self.send_command("LST ALMAF:;")
        report_blocks = re.split(r'(?m)^\+\+\+\s+', resposta)[1:]  # o primeiro split antes do 1º +++ é lixo

        all_records: List[Dict] = []
        for block in report_blocks:
            # extrai device e timestamp do cabeçalho do bloco
            header, _, rest = block.partition('%%')  # header vai até '%%'
            m = re.match(r'(\S+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', header.strip())
            report_device, report_time = (m.group(1), m.group(2)) if m else (None, None)

            # 2) Separa cada “ALARM …” como um sub-bloco
            #    usamos lookahead para não perder o “ALARM” no split
            alarm_blocks = re.split(r'(?m)^(?=ALARM\s+\d+)', rest.strip())

            for alarm in alarm_blocks:
                if not alarm.strip().startswith('ALARM'):
                    continue

                rec: Dict = {
                    'report_device': report_device,
                    'report_time': report_time
                }

                lines = alarm.splitlines()

                # 3) Linha principal “ALARM  290352  Fault  Critical  SRAN  26818  Running”
                first = lines[0].split()
                # first = ['ALARM', id, class, severity, domain, code, category]
                rec.update({
                    'alarm_id':       first[1],
                    'alarm_class':    first[2],
                    'severity':       first[3],
                    'domain':         first[4],
                    'code':           first[5],
                    'category':       first[6],
                })

                # 4) As demais linhas são no formato “chave = valor”
                for ln in lines[1:]:
                    m = re.match(r'^\s*([\w\s\-/\(\)]+?)\s*=\s*(.*)$', ln)
                    if not m:
                        continue
                    key, val = m.group(1).strip(), m.group(2).strip()

                    # normaliza nome de coluna: minuscula, underline
                    col = key.lower().replace(' ', '_')
                    rec[col] = val

                all_records.append(rec)

        # 5) Converte lista de dicts em DataFrame, preenchendo com NaN os campos ausentes
        df = pd.DataFrame(all_records)
        return df


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
        try:
            regnestring = f'REG NE:NAME={ne_name};'
            result = self.send_command(regnestring)
            
            if "RETCODE = 0  Success" not in result:
                raise Exception(f"Failed to connect to {ne_name}. Error: {result}")
        except Exception as e:
            print(e)


    def _unreg_ne(self, ne_name):
        """Unregister NE with given name (internal use only)"""
        try:
            regnestring = f'UNREG NE:NAME={ne_name};'
            result = self.send_command(regnestring)
            
            if "RETCODE = 0  Success" not in result:
                raise Exception(f"Failed to unregister {ne_name}. {result}")
        except Exception as e:
            print(e)
    
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


if __name__ == "__main__":
    
    df_report = pd.DataFrame()
    Mae = NBI()
    Mae.connect()
    Sites = Mae.get_ne_list()
    Sites = {key: value for key, value in Sites.items() if value['type'].startswith("BTS")}

    for site in Sites: 
        print(f"Site {site} is of type BTS.")
        
        try:
            Mae.select_ne(site)
            df = Mae.get_alarms()
            df_report = pd.concat([df_report, df], ignore_index=True)
        except:
            pass



df_report = df_report.drop_duplicates()
df_report.to_excel('reportCadu.xlsx', index=False)
df_report.shape



