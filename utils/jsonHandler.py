import os
import json

def read_json_files(folder_path):
    json_data = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_data[filename] = []
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                for line in file:
                    try:
                        line_data = json.loads(line)
                        json_data[filename].append(line_data)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line: {line}")
    
    return json_data


