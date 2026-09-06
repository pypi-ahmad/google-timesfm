"""Export API contracts without connecting to PostgreSQL or loading a model."""

import argparse
import json
from pathlib import Path

from .api import create_app


def main():
  """Write the model-free FastAPI OpenAPI contract to the requested path."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  destination = parser.parse_args().output
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(json.dumps(create_app().openapi(), indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
