
from backend.app.datasets.loader import DatasetLoader


def main():

    loader = DatasetLoader()

    try:

        dataset = loader.load(
            "data/invalid_dataset.json"
        )

        print("Dataset loaded successfully!")

    except Exception as error:

        print("Dataset validation failed!")

        print("Error:", error)


if __name__ == "__main__":
    main()
