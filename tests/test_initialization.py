import json
from pathlib import Path
from typer.testing import CliRunner

from active_learning import __app_name__, __version__, cli

runner = CliRunner()

home = Path.home()


def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} v{__version__}\n" in result.stdout
    
def test_init_input():
    """Test initialization with path as typed input"""
    init_path = home.joinpath(".app_params_test1.json")
    assert not Path(init_path).exists()
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    assert Path(init_path).exists()
    Path(init_path).unlink(missing_ok=True)

def test_init_param():
    """Test initialization with path input"""
    init_path = home.joinpath(".app_params_test2.json")
    assert not Path(init_path).exists()
    result = runner.invoke(cli.app, ["init", "--param-path", f"{str(init_path)}"])
    assert result.exit_code == 0
    assert Path(init_path).exists()
    Path(init_path).unlink(missing_ok=True)
    assert not Path(init_path).exists()
    result = runner.invoke(cli.app, ["init", "-pp", f"{str(init_path)}"])
    assert result.exit_code == 0
    assert Path(init_path).exists()
    Path(init_path).unlink(missing_ok=True)
    
def test_init_existing():
    """Test initialization if file is existing - will be overwritten"""
    init_path = home.joinpath(".app_params_test3.json")
    assert not Path(init_path).exists()
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    assert Path(init_path).exists()
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    assert Path(init_path).exists()
    Path(init_path).unlink(missing_ok=True)

def test_init_invalid_path():
    """Test initialization if invalid path passed"""
    init_path = home.joinpath("/nonesistent/.app_params_test.json")
    assert not Path(init_path).exists()
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 1
    assert 'Creating parameter file failed with "parameter value error"' in result.stdout
    assert not Path(init_path).exists()
    
def test_init_file_nonempty():
    """Test if file generated is not empty"""
    init_path = home.joinpath(".app_params_test4.json")
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    params = json.loads(init_path.read_text())
    assert isinstance(params, dict)
    assert len(params) == 13
    Path(init_path).unlink(missing_ok=True)

def test_train_without_init():
    """ """
    result = runner.invoke(cli.app, ["train"])
    assert result.exit_code == 1
    assert 'Parameter file not found' in result.stdout

def test_register_simple():
    init_path = home.joinpath(".app_params_test5.json")
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    params = json.loads(init_path.read_text())
    assert params["valid_n"] == 0.1
    result = runner.invoke(cli.app, ["register", "valid_n=0.3"], input=str(init_path))
    assert result.exit_code == 0
    params = json.loads(init_path.read_text())
    assert params["valid_n"] == 0.3
    Path(init_path).unlink(missing_ok=True)

def test_register_complex():
    init_path = home.joinpath(".app_params_test6.json")
    result = runner.invoke(cli.app, ["init"], input=str(init_path))
    assert result.exit_code == 0
    params = json.loads(init_path.read_text())
    assert params["optimizer"].get("lr", None) is None
    result = runner.invoke(cli.app, ["register", "optimizer.lr=0.3"], input=str(init_path))
    assert result.exit_code == 0
    params = json.loads(init_path.read_text())
    assert params["optimizer"].get("lr", None) == 0.3
    Path(init_path).unlink(missing_ok=True)
    