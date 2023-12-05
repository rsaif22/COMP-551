from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Replace 'dataset-owner/dataset-name' with the actual dataset path on Kaggle
def download_dataset(dataset_path):
    api.dataset_download_files(dataset_path, path='./', unzip=True)
