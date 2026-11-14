
from pathlib import Path

from backend.app.datasets.loader import DatasetLoader


def main():

    loader = DatasetLoader()

    project_root = Path(__file__).parent

    dataset_files = [
        project_root / "data" / "sample_dataset.json",
        project_root / "data" / "sample_dataset.jsonl"
    ]

    for file_path in dataset_files:

        print("\n" + "=" * 60)

        print(f"Loading dataset: {file_path.name}")

        print("=" * 60)

        dataset = loader.load(str(file_path))

        print(f"Dataset Name: {dataset.dataset_name}")

        print(f"Description: {dataset.description}")

        print(f"Total Questions: {len(dataset.samples)}")

        for sample in dataset.samples:

            print("\nQuestion ID:", sample.id)

            print("Question:", sample.question)

            print("Expected Answer:", sample.expected_answer)

            print("Category:", sample.category)

        print("\nDataset loaded and validated successfully!")

    print("\nAll dataset tests completed!")


if __name__ == "__main__":
    main()
