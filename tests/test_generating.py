import json
from pathlib import Path
import shutil
import pytest
from typer.testing import CliRunner

from active_learning import __app_name__, __version__, cli

runner = CliRunner()

home = Path.home()
init_path = home.joinpath(".app_params_test.json")
labeled_path = "data/sample_dataset/labeled"
unlabeled_path = "data/sample_dataset/unlabeled"
to_annotate_path = "data/sample_dataset/to_annotate_test"
annotated_path = "data/sample_dataset/annotated_2"

@pytest.fixture(scope="module")
def initialization():
    runner.invoke(cli.app, ["init"], input=str(init_path))
    yield 
    Path(init_path).unlink(missing_ok=True)

@pytest.fixture(scope="module")
def initialization_with_classes():
    runner.invoke(cli.app, ["init"], input=str(init_path))
    with open(init_path, 'r') as f:
        existing_data = json.load(f)
    existing_data.update({'classes': 3})
    with open(init_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    yield existing_data
    Path(init_path).unlink(missing_ok=True)

@pytest.fixture(scope="module")
def initialization_with_labeled():
    runner.invoke(cli.app, ["init"], input=str(init_path))
    with open(init_path, 'r') as f:
        existing_data = json.load(f)
    existing_data.update(
        {'labeled_dataset': labeled_path})
    with open(init_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    yield existing_data
    Path(init_path).unlink(missing_ok=True)

def test_no_dataset(initialization):
    """Dataset path is a compulsory parameter"""
    result = runner.invoke(cli.app, ["generate"])
    assert result.exit_code == 1
    assert 'Generating failed with "error loading dataset"' in result.stdout

def test_unknown_classes(initialization):
    """Generating samples requires knowledge of number of classes"""
    result = runner.invoke(cli.app, ["generate", "--path", labeled_path])
    assert result.exit_code == 1
    assert 'Generating failed with "parameter value error"' in result.stdout
    
def test_no_outpath(initialization_with_classes):
    """Path to save chosen samples is required"""
    result = runner.invoke(cli.app, ["generate", "--path", unlabeled_path, "--random"])
    assert result.exit_code == 1
    assert 'Generating failed with "parameter value error"' in result.stdout

def test_generating_random(initialization_with_classes):
    """Samples can be chosen at random"""
    assert not Path(to_annotate_path).exists()
    result = runner.invoke(cli.app, ["generate", "--path", unlabeled_path, "--random", "--outpath", to_annotate_path])
    assert result.exit_code == 0
    assert Path(to_annotate_path).exists()
    assert Path(to_annotate_path).is_dir()
    assert any(Path(to_annotate_path).iterdir())
    shutil.rmtree(Path(to_annotate_path))

def test_generating_fast(initialization_with_classes):
    """Samples can be chosen with fast method - largest margin"""
    assert not Path(to_annotate_path).exists()
    result = runner.invoke(cli.app, ["generate", "--path", unlabeled_path, "--fast", "--outpath", to_annotate_path])
    assert result.exit_code == 0
    assert Path(to_annotate_path).exists()
    assert Path(to_annotate_path).is_dir()
    assert any(Path(to_annotate_path).iterdir())
    assert 'Largest Margin' in result.stdout
    shutil.rmtree(Path(to_annotate_path))
    
def test_generating_normal(initialization_with_classes):
    """Samples can be chosen with normal method - monte carlo dropout"""
    assert not Path(to_annotate_path).exists()
    result = runner.invoke(cli.app, ["generate", "--path", unlabeled_path, "--outpath", to_annotate_path])
    assert result.exit_code == 0
    assert Path(to_annotate_path).exists()
    assert Path(to_annotate_path).is_dir()
    assert any(Path(to_annotate_path).iterdir())
    assert 'MC Dropout' in result.stdout
    shutil.rmtree(Path(to_annotate_path))
    
def test_merge(initialization_with_labeled):
    assert Path(labeled_path).exists()
    assert Path(labeled_path).is_dir()
    initial_elements = len(list(Path(labeled_path).glob('**/*'))) - len(list(Path(labeled_path).glob('*')))
    new_elements = len(list(Path(annotated_path).glob('**/*'))) - len(list(Path(annotated_path).glob('*')))
    result = runner.invoke(cli.app, ["merge", annotated_path])
    assert result.exit_code == 0
    final_elements = len(list(Path(labeled_path).glob('**/*'))) - len(list(Path(labeled_path).glob('*')))
    assert initial_elements + new_elements == final_elements
