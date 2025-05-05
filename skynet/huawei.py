import time
import datetime
from .setup import *



MAXUSERS = {
    "1800_15": 150, 
    "2600_20": 250, # entre 200 e 800
    "2600_10": 50,
    "2100_10": 50,
}
VELOCIDADE = 20 # segundos. Tempo de espera entre comandos
CONFIDENCE = 0.95
TRIALS = 30
TILT_RANGE = (-15,15)
AZIMUTH_RANGE = (-30,30)
RS_RANGE = (25, 95)
RS_STEP = 10
REPEAT = 1 # quantas x repete a probe de usuarios e pega o maior valor

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
                       "26-4C"]
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
                       "18-4C"]
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
                       "21-4C"]
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

# Função que atua na rede, consultando a quantidade de usuarios conectados no momento.
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

def AjusteDePotencia(connection, sentido, cell, localcellid):
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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if sentido == 'reduzir':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        RsAtual = ParserRsSignal(connection.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RsAtual - RS_STEP >= RS_RANGE[0]:
            RsNovo = RsAtual - RS_STEP
            connection.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
        else:
            log_entry = f"[{timestamp}] {cell} ja tem mais que {MaxUsers(cell)} usuarios. RSignal já no mínimo = {RS_RANGE[0]} dBm\n"

    elif sentido == 'aumentar':
        # MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};
        RsAtual = ParserRsSignal(connection.send_command(f"LST PDSCHCFG:LOCALCELLID={localcellid};"))
        if RsAtual + RS_STEP <= RS_RANGE[1]:
            RsNovo = RsAtual + RS_STEP
            connection.send_command(f"MOD PDSCHCFG:LOCALCELLID={localcellid},REFERENCESIGNALPWR={RsNovo};")
            log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal ajustado = {RsNovo} dBm\n"
        else:
            log_entry = f"[{timestamp}] {cell} tem menos que {MaxUsers(cell)} usuarios. RSignal já no máximo = {RS_RANGE[1]} dBm\n"
    else:
        log_entry = f"[{timestamp}] ERRO NO AJUSTE DE RS\n"

    print(log_entry)
    with open(LOG_RESULTS, "a") as f:
        f.write(log_entry)
