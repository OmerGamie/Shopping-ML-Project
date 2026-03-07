import os 
import subprocess

def download_kaggle_dataset(dataset):
    """
    Downloads and unzips a Kaggle dataset using kaggle CLI
    
    Parameters:
    -----------
    dataset: str
    The kaggle dataset identifier in the format 'username/dataset-name'
    
    Behavior:
    ---------
    - Creates the folder data/raw if it does not exist.
    - Downloads the speciified kaggle dataset to 'data/raw'.
    - Automatically unzips the dataset
    
    Requirements:
    -------------
    - Kaggle CLI must be installed ('pip install kaggle')
    - Kaggle API token must be placed at '~/.kaggle/kaggle.json'   
    """
    # Create the raw data folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Build the kaggle CLI command
    cmd = f"kaggle datasets download -d {dataset} -p data/raw --unzip"
    
    # Run the command in the system shell
    subprocess.run(cmd, shell=True)

# Entry Point: Run only if this script executed directly    
if __name__ =="__main__":
    # Specifiy the dataset identifier from kaggle
    download_kaggle_dataset("shree0910/online-vs-in-store-shopping-behaviour-dataset")
    