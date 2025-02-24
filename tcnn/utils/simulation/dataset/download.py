import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def download(url, save_dir=os.path.join("./", "data")):
    """
    Downloads a file from the given URL and saves it to the specified directory.

    Args:
        url (str): The URL of the file to download.
        save_dir (str, optional): The directory to save the downloaded file. Defaults to './data'.

    Returns:
        str: The file path of the downloaded file if successful, None otherwise.
    """
    os.makedirs(save_dir, exist_ok=True)

    file_name = url.split("/")[-1]
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        logging.info(f"{file_name} exists!")
        return file_path
    logging.info(f"downloading {file_name} from {url}")
    response = requests.get(url)
    if response.ok:
        total = int(response.headers.get("content-length"))
        print(f"total size: {total}")
        with open(file_path, "wb") as f, tqdm(
            desc=file_name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            size = f.write(response.content)
            bar.update(size)
        logging.info(f"download  {file_name} from {url} successfully!")
        return file_path
    else:
        print(f"Fail to download  {file_name} from {url}")
        return None
