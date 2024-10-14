import json
import signal
import threading
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from application import Application



def get_execution_arguments() -> Namespace:
    parser = ArgumentParser(
        prog='main.py',
        description='Starts the application',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        help='Path to the configuration file',
        type=str,
        default='config/configuration.json'
    )
    return parser.parse_args()


def get_config_path(args: Namespace) -> str:
    return args.config

def get_config_data(config_path:str):
    configuration_file = open(config_path, 'r')
    config_data = json.load(configuration_file)
    configuration_file.close()
    Application(config_data)
