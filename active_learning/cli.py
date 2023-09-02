import json
import shutil
import typer

from pathlib import Path
from typing import Any, List, Optional

from active_learning import __app_name__, __version__, ERRORS, SUCCESS, FILE_ERROR, PARAM_ERROR, config, params

app = typer.Typer()

@app.command()
def init(
    param_path: str = typer.Option(
        str(params.DEFAULT_PARAM_FILE_PATH),
        "--param-path",
        "-pp",
        prompt="parameter path",
    ),
) -> None:
    """Initialize the parameters file"""
    app_init_error = config.init_app(param_path)
    if app_init_error:
        typer.secho(
            f'Creating config file failed with "{ERRORS[app_init_error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    param_init_error = params.init_param_file(Path(param_path))
    if param_init_error:
        typer.secho(
            f'Creating parameter file failed with "{ERRORS[param_init_error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    else:
        typer.secho(f"The param file is {param_path}. Change it manually or using the app.", fg=typer.colors.GREEN)

def get_active_learning_handler():
    from active_learning.active import ActiveLearningHandler
    if config.CONFIG_FILE_PATH.exists():
        param_path = params.get_param_path(config.CONFIG_FILE_PATH)
    else:
        typer.secho(
            f'Config file not found. Please, run "{__app_name__} init"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if param_path.exists():
        return ActiveLearningHandler(param_path)
    else:
        typer.secho(
            f'Parameter file not found. Please, run "{__app_name__} init"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

@app.command()
def train(
    resume: int = typer.Option(0, "--resume", "-r", min=0, help="Index of epoch to resume training from. Use in case of recovering from app crash"),
    cont: bool = typer.Option(False, "--continue", "-c", help="Use if training to be continued on last iteration model"),
    labeled: str = typer.Option(None, "--path", "-p", help="Path to the labeled dataset for training")
) -> None:
    """Start or continue training the model used for choosing the subset of data"""
    try:
        al_handler = get_active_learning_handler()
        if labeled:
            al_handler.al_iter.unlabeled_dataset = labeled
        if resume > 0 and cont:
            result = PARAM_ERROR
        result = al_handler.train(resume, cont)
    except json.decoder.JSONDecodeError:
        result = PARAM_ERROR

    if result != SUCCESS:
        typer.secho(
            f'Running iteration failed with "{ERRORS[result]}"', fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            f"""Iteration run from {'beggining' if resume == 0 else 'epoch ' + str(resume) }""",
            fg=typer.colors.GREEN,
        )

@app.command()
def generate(
    fast: bool = typer.Option(False, "--fast", "-f", help="Method for generating the subset"), 
    random: bool = typer.Option(False, "--random", "-r", help="Generate subset randomly"), 
    quantity: float = typer.Option(0.1, "--quantity", "-q", help="Number of samples to be generated for labeling or percentage of whole dataset"),
    unlabeled: str = typer.Option(None, "--path", "-p", help="Path to the unlabeled dataset to sample for labeling"),
    out_path: str = typer.Option(None, "--outpath", "-o", help="Path to the folder where sampled images will be moved (or copied) to"),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy the sampled images. By default they are moved"), 
) -> None:
    """Start or continue training the model used for choosing the subset of data"""
    if random:
        typer.secho(
            f"Random selection chosen. No heuristic used." ,
            fg=typer.colors.YELLOW,
        )
    try:
        al_handler = get_active_learning_handler()
        if unlabeled:
            al_handler.al_iter.unlabeled_dataset = unlabeled
        if out_path:
            al_handler.al_iter.to_annotate = out_path
        al_handler.al_iter = al_handler.al_iter._replace(copy=copy)
        result = al_handler.generate(fast, random, quantity)
    except json.decoder.JSONDecodeError:
        result = PARAM_ERROR

    if result != SUCCESS:
        typer.secho(
            f'Running iteration failed with "{ERRORS[result]}"', fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            f"""Samples generated """
            f"""with method: {"Largest Margin" if fast else "MC Dropout"}""",
            fg=typer.colors.GREEN,
        )

def _register(p):
    parsed_dict = dict(item.split('=') for item in p)
    if config.CONFIG_FILE_PATH.exists():
        param_path = params.get_param_path(config.CONFIG_FILE_PATH)
    else:
        raise FILE_ERROR
    if param_path.exists():
        return params.update_param_file(param_path, parsed_dict)
    else:
        raise PARAM_ERROR
    
@app.command()
def register(
    parameters: List[str] 
) -> None:
    """Register parameter by passing string of form key=value """
    result = _register(parameters)
    if result != SUCCESS:
        typer.secho(
            f'Running iteration failed with "{ERRORS[result]}"', fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        typer.secho("",
            fg=typer.colors.GREEN,
        )

def _merge(source):
    if config.CONFIG_FILE_PATH.exists():
        param_path = params.get_param_path(config.CONFIG_FILE_PATH)
    with param_path.open("r", encoding="UTF-8") as f:
        param_dict = json.load(f)
    dest = Path(param_dict["labeled_dataset"])
    for p in Path(source).rglob('*'):
        relative_path = p.relative_to(source)
        destination_path = dest / relative_path

        if p.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination_path)
    return SUCCESS
    
        
@app.command()
def merge(
    path: str 
) -> None:
    """Merge the new labeled subset into training data"""
    result = _merge(path)
    if result != SUCCESS:
        typer.secho(
            f'Running iteration failed with "{ERRORS[result]}"', fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        typer.secho("",
            fg=typer.colors.GREEN,
        )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return