from decouple import Config, RepositoryEnv
import os
# Define o caminho absoluto do diretório atual (onde setup.py está)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.ini")

# Inicializa o Config utilizando o caminho absoluto
config = Config(RepositoryEnv(SETTINGS_PATH))

#config = Config(RepositoryEnv('skynet/settings.ini'))
TeclaEnter = "\r\n"
LOG_FILE = "telnet_commands.log"
LOG_RESULTS = "results.log"
LOG_EVENTO = "eventos.log"
MAEHOSTNAME = config('MAEHOSTNAME')
MAEPORT = config('MAEPORT')
LOGIN = config('LOGIN')
PASSWORD = config('PASSWORD')
ZAPNUMBER = config('ZAPNUMBER')
