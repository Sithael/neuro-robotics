import click
import stable_baselines3
from baselines import BaselineCore
from utils.common import constants
from utils.common import methods


def fetch_baselines_model(model: str):
    try:
        model = getattr(stable_baselines3, model)
    except ImportError:
        raise SystemError(f"Pipeline could not fetch the model: {model}")
    return model


def evaluate():
    pass


@click.command()
@click.option("--settings", default="baseline", help="Metadata yaml file id")
def launch(settings):
    settings_directory = constants.SETTINGS_DIR / f"{settings}.yml"
    metadata = methods.load_yaml(settings_directory)
    model = fetch_baselines_model(metadata["baseline"]["model"])
    baseline_core = BaselineCore(metadata, model)
    baseline_core.train_model()


if __name__ == "__main__":
    launch()
