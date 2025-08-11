import os

input_folder = "data/tickets"
output_file = "corpus.txt"

with open(output_file, "w", encoding="utf-8") as out_f:
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".txt"):
            continue
        
        file_path = os.path.join(input_folder, fname)
        with open(file_path, "r", encoding="utf-8") as in_f:
            content = in_f.read()
        
        # Replace multiple whitespace/newlines with single space
        # This makes the whole ticket one line for MLM purposes
        cleaned = " ".join(content.strip().split())
        
        if cleaned:  # Avoid empty lines
            out_f.write(cleaned + "\n")

print(f"Corpus created: {output_file}")
















"""
from a2rchi.utils.config_loader import load_config_file
import os

config = load_config_file()
data_path = config["DATA_PATH"]
corpus_file = os.path.join(data_path, "corpus.txt")

with open(corpus_file, "w", encoding="utf-8") as outfile:
    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if os.path.isdir(dir_path) and dir_name != "vstore":
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(f"\n--- {file_name} ---\n")
                        outfile.write(infile.read())
                        outfile.write("\n")
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
"""
