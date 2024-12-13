import os
import shutil

# Specify the directory you want to clear

def clean_directory(dir_name):

    # Loop through all the files in the directory
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        try:
            if os.path.isfile(file_path):  # Check if it's a file
                os.unlink(file_path)       # Delete the file
            elif os.path.isdir(file_path):  # Optional: Remove directories if needed
                shutil.rmtree(file_path)    # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
