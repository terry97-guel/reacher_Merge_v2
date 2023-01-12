import csv
import os


class CSVLogger:
    def __init__(self, path):
        self.path = str(path)
        if os.path.exists(self.path):
            os.remove(path)
        
        self.clear = True
        
    def create(self,data:dict):
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()

    def log(self, data:dict):
        data = {f"{key}":float(value) for key, value in data.items()} 
        
        if self.clear:
            self.create(data)
            self.clear = False
        
        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writerow(data)

from pathlib import Path
import os
import shutil


def ask_and_make_folder(path:Path):
    if path.exists():
        print(f"Save Directory already exists! Want to delete {path.__str__()}?")
        print("y for yes, n for no")
        delete_folder = input()
        if delete_folder == 'y':
            shutil.rmtree(path.absolute())
            Path.mkdir(path, parents=True)
        else:
            print("Exitting...")
            exit(1)
    else:    
        Path.mkdir(path, parents=True)