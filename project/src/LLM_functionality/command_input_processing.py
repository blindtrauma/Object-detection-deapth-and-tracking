from src.functionalities.execution import match_command
from src.summon_logic.audio_processing import process_voice_command
def terminal_task():
    textt = process_voice_command()
    a, b = match_command(textt)
    print(a, b)
    