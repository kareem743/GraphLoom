
def load_data(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def clean_data(data):
    return [line.strip() for line in data if line.strip()]

def tokenize_line(line):
    return line.split()

def tokenize_data(data):
    return [tokenize_line(line) for line in data]

def flatten(nested_list):
    if not nested_list:
        return []
    if isinstance(nested_list[0], list):
        return flatten(nested_list[0]) + flatten(nested_list[1:])
    return [nested_list[0]] + flatten(nested_list[1:])

def count_tokens(flat_tokens):
    return len(flat_tokens)

class FileProcessor:
    def __init__(self, path):
        self.path = path

    def process(self):
        raw = load_data(self.path)
        cleaned = clean_data(raw)
        tokenized = tokenize_data(cleaned)
        flat = flatten(tokenized)
        return count_tokens(flat)

def run_pipeline(file_path):
    processor = FileProcessor(file_path)
    return processor.process()


