from huggingface_hub import HfApi

api = HfApi()

# upload content from folder
data_folder = "/Users/idegen/works/corrclust-validation/parquet_data"

dataset_name = "idegen/csts"

print(f"Uploading from {data_folder} to {dataset_name}")

# upload the exploratory and confirmatory folders and all subfolders
api.upload_large_folder(
    folder_path=data_folder,
    repo_id=dataset_name,
    repo_type="dataset",
    ignore_patterns=[".*", ".py"],  # ignore hidden files and python
)

print("Hurray upload has completed")
