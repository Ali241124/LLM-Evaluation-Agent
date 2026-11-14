
import json

from pathlib import Path
from pydantic import ValidationError

from backend.app.schemas.dataset import (
    DatasetFile,
    DatasetSample
)


class DatasetLoader:
    """
    Loads and validates JSON and JSONL datasets.
    """

    def load(self, file_path: str) -> DatasetFile:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a file, received: {file_path}"
            )

        extension = path.suffix.lower()

        try:

            if extension == ".json":
                return self._load_json(path)

            elif extension == ".jsonl":
                return self._load_jsonl(path)

            else:
                raise ValueError(
                    f"Unsupported dataset format: {extension}"
                )

        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(
                f"Invalid dataset file: {error}"
            ) from error

    def _load_json(self, path: Path) -> DatasetFile:
        """
        Load a structured JSON dataset.
        """

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return DatasetFile.model_validate(data)

    def _load_jsonl(self, path: Path) -> DatasetFile:
        """
        Load a JSONL dataset.
        """

        samples = []

        with path.open("r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    sample = DatasetSample.model_validate(data)

                    samples.append(sample)

                except (json.JSONDecodeError, ValidationError) as error:
                    raise ValueError(
                        f"Invalid JSONL record at line "
                        f"{line_number}: {error}"
                    ) from error

        return DatasetFile(
            dataset_name=path.stem,
            description="Loaded from JSONL file",
            samples=samples
        )
