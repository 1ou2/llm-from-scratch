# ----------------------------------------------------------------------
# Load configuration from file
# ----------------------------------------------------------------------
import configparser
import ast
import os

def load_config(config_file="config.txt"):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Convert string values to appropriate Python types
    def parse_value(value):
        try:
            # Try to evaluate as literal (for boolean, None, etc)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it fails, return as string
            return value

    # Create configuration dictionaries
    GPT_CONFIG = {
        key: parse_value(value)
        for key, value in config['model'].items()
    }
    
    HYPERS = {
        key: parse_value(value)
        for key, value in config['hypers'].items()
    }
    
    #
    FILES = {
        key: parse_value(value)
        for key, value in config['files'].items()
    }

    TRAINING = {
        key: parse_value(value)
        for key, value in config['training'].items()
    }
    
    return GPT_CONFIG, HYPERS, FILES, TRAINING

class LogPrinter:
    def __init__(self, log_file):
        # check if parent directory exists
        parent_dir = os.path.dirname(log_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.log_file = open(log_file,"a")

    def log_print(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()