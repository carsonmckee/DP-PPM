from pathlib import Path

def clear_csv_files(folder_path):
    """
    Deletes all .csv files in the given folder
    and prints the number of files deleted.
    
    :param folder_path: Path to the folder containing CSV files
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    count = 0

    for csv_file in folder.glob("*.csv"):
        try:
            csv_file.unlink()  # Delete file
            count += 1
        except Exception:
            pass  # Silently ignore errors

    print(f"{count} CSV file(s) deleted.")


if __name__ == "__main__":

    clear_csv_files('results/ar')
    clear_csv_files('results/normal')