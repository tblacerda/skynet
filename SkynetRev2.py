import pandas as pd
import schedule
import datetime
import time
# imports locais
from skynet.NBI import NBI
from skynet.BayesianOptimization import maximize_users, AjustForBestPos
from skynet.huawei import primary_cell, secondary_cell, MaxUsers, FunctionUsers, AjusteDePotencia
from skynet.tstudent import AnaliseEstatistica
import threading

DURATION_HOURS = 6 # 7 dias
MONITORAR_SITES = [ 'SR-CE05CW', 'SR-OLSGJ0']
       

if __name__ == "__main__":

    # checa se a conexão com Mae está ativa
    Mae = NBI()
    # se a conexão nao estiver ativa, conecta
    Mae.connect()
    Mae.SITES = MONITORAR_SITES
    # Schedule a job to run every 5 minutes
    schedule.every(1).minute.do(Mae.update_usuarios_sites)
    schedule.every(10).minutes.do(AnaliseEstatistica)
    # dos sites indicados
    # realizar otimizacao
    


    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)  # Check every second
        



    df_report = pd.DataFrame()
    Mae = NBI()
    Mae.connect()
    # Calcular horários de início e fim



