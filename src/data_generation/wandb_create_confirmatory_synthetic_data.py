from src.data_generation.wandb_create_synthetic_data import create_datasets
from src.utils.configurations import WandbConfiguration, CONFIRMATORY_SYNTHETIC_DATA_DIR

if __name__ == "__main__":
    # confirmatory data creation
    create_datasets(data_dir=CONFIRMATORY_SYNTHETIC_DATA_DIR, seed=1905,
                    wand_db_project_name=WandbConfiguration.wandb_confirmatory_project_name, n=30,
                    tag='30_ds_confirmatory_creation')
