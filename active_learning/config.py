"""This module provides the config functionality."""

import configparser
from pathlib import Path

import typer

from active_learning import (
    __app_name__, SUCCESS, DIR_ERROR, FILE_ERROR, PARAM_ERROR
)

CONFIG_DIR_PATH = Path(typer.get_app_dir(__app_name__))
CONFIG_FILE_PATH = CONFIG_DIR_PATH / "config.ini"

def init_app(param_path: str) -> int:
    """Initialize the application."""
    config_code = _init_config_file()
    if config_code != SUCCESS:
        return config_code
    parameters_code = _write_parameters(param_path)
    if parameters_code != SUCCESS:
        return parameters_code
    return SUCCESS

def _init_config_file() -> int:
    try:
        CONFIG_DIR_PATH.mkdir(exist_ok=True)
    except OSError:
        return DIR_ERROR
    try:
        CONFIG_FILE_PATH.touch(exist_ok=True)
    except OSError:
        return FILE_ERROR
    return SUCCESS

def _write_parameters(param_path: str) -> int:
    config_parser = configparser.ConfigParser()
    config_parser["General"] = {"param_path": param_path}
    try:
        with CONFIG_FILE_PATH.open("w") as file:
            config_parser.write(file)
    except OSError:
        return PARAM_ERROR
    return SUCCESS