import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from active_learning import __app_name__, __version__, cli

runner = CliRunner()

home = Path.home()
init_path = home.joinpath(".app_params_test.json")

@pytest.fixture(scope="module")
def initialization():
    runner.invoke(cli.app, ["init"], input=str(init_path))
    with open(init_path, 'r') as f:
        existing_data = json.load(f)
    existing_data.update({'epochs': 2})
    with open(init_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    yield existing_data
    Path(init_path).unlink(missing_ok=True)

def test_conflicting_train_params(initialization):
    """Resume and continue cannot be used together"""
    result = runner.invoke(cli.app, ["train", "--resume", "7", "--continue" ])
    assert result.exit_code == 1
    assert 'Running iteration failed with "parameter value error"' in result.stdout

def test_no_dataset(initialization):
    """Dataset path is a compulsory parameter"""
    result = runner.invoke(cli.app, ["train"])
    assert result.exit_code == 1
    assert 'Running iteration failed with "error loading dataset"' in result.stdout

def test_parameter_dataset(initialization):
    """Dataset path is a compulsory parameter"""
    assert not home.joinpath("checkpoints/epoch-1.pth").exists()
    result = runner.invoke(cli.app, ["train", "--path", "data/sample_dataset/labeled"])
    assert result.exit_code == 0
    assert 'Iteration run from beggining' in result.stdout
    assert home.joinpath("checkpoints/epoch-1.pth").exists()

def test_resume(initialization):
    """Resume training from specific epoch"""
    assert not home.joinpath("checkpoints/epoch-1.pth").exists()
    result = runner.invoke(cli.app, ["train", "--path", "data/sample_dataset/labeled", "--resume", "1"])
    assert result.exit_code == 0
    assert 'Iteration run from epoch 0' in result.stdout
    assert home.joinpath("checkpoints/epoch-1.pth").exists()
    
def test_continue(initialization):
    """Resume training from last epoch"""
    previous_epoch_tstmp = home.joinpath("checkpoints/epoch-0.pth").stat().st_mtime
    result = runner.invoke(cli.app, ["train", "--path", "data/sample_dataset/labeled", "--continue"])
    assert result.exit_code == 0
    assert home.joinpath("checkpoints/epoch-0.pth").stat().st_mtime != previous_epoch_tstmp
    assert 'Iteration run from beggining' in result.stdout