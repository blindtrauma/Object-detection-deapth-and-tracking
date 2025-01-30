import re
from src.utils.load_func import load_json_file
from src.functionalities.functionalities_definition import ai_image_analysis_and_summary,ai_studio,annotation_from_experts,ar_sign_reading,job_management_and_safety_checks, qr_bar_code_scanner


# file_name = r'functionalities/functionality.json'
# functionality_file = load_json_file(file_name)
# user_commands = functionality_file.get("commands", {})

def match_command(text):
    file_name = r'src/functionalities/functionality.json'
    functionality_file = load_json_file(file_name)
    user_commands = functionality_file.get("commands", {})
    for command_name, command_info in user_commands.items():
        pattern = command_info["pattern"]
        print(text, "this is the text input")
        if re.search(pattern, text, re.IGNORECASE):
            print(f"Matched command '{command_name}' with function '{command_info['function']}'")
            return command_name, command_info['function']
    return None, None








