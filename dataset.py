import kaggle 

kaggle.api.authenticate()

kaggle.api.dataset_download_files('oktayrdeki/houses-in-london', path='.', unzip=True)


