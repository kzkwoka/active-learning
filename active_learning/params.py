import configparser
import json
from pathlib import Path

from active_learning import PARAM_ERROR, SUCCESS

DEFAULT_PARAM_FILE_PATH = Path.home().joinpath(
    ".app_params.json"
)

DEFAULT_CHECKPOINT_FILE_PATH = Path.home().joinpath(
    f"checkpoints/epoch-nn.pth"
)

DEFAULT_PARAMS = dict(
    model="efficientnet_v2_s",
    weights="EfficientNet_V2_S_Weights.DEFAULT",
    valid_n = 0.1,
    
    labeled_dataset = "",
    unlabeled_dataset = "",
    to_annotate = "",
    
    loss = dict(name="CrossEntropyLoss"),
    optimizer = dict(name="Adam"),
    scheduler = dict(name="ReduceLROnPlateau"),
    
    batch_size = dict(train=64),
    epochs=10,
    
    checkpoint_path = str(DEFAULT_CHECKPOINT_FILE_PATH),
    checkpoint_every = 1
)

def get_param_path(config_file: Path) -> Path:
    """Return the current path to the parameter file."""
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    return Path(config_parser["General"]["param_path"])

def init_param_file(param_path: Path) -> int:
    """Initialize the parameter file."""
    try:
        with param_path.open("w", encoding="UTF-8") as f:
            json.dump(DEFAULT_PARAMS, f, indent=4)
        return SUCCESS
    except OSError:
        return PARAM_ERROR


def recursive_update(d, update_with):
    for key, value in update_with.items():
        keys = key.split('.')
        current_dict = d
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        try:
            current_dict[keys[-1]] = eval(value)
        except NameError:
            current_dict[keys[-1]] = value
    return d

def update_param_file(param_path: Path, update_dict: dict) -> int:
    try:
        with param_path.open("r", encoding="UTF-8") as f:
            params = json.load(f)

        params = recursive_update(params, update_dict)

        with param_path.open("w", encoding="UTF-8") as f:
            json.dump(params, f, indent=4)
            return SUCCESS
    except OSError:
        return PARAM_ERROR