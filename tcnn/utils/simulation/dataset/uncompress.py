import zipfile


def unzip_file(file_path, output_dir):
    """
    Unzips a file to the specified output directory.

    Parameters:
        file_path (str): The path to the zip file.
        output_dir (str): The directory where the contents of the zip file will be extracted.
    """
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"unzip {file_path} to {output_dir} successfully!")
