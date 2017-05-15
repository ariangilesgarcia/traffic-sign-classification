import os
import glob
import json
import random

# Read csv file
dataset_path = '/home/arian/Documents/proyecto-integrador/data/dataset-v4/'

classes_folders = glob.glob(os.path.join(dataset_path, '*'))

json_data = {}

for class_folder in classes_folders:
    class_id, label = os.path.basename(class_folder).split('-', maxsplit=1)

    json_data[class_id] = {}
    json_data[class_id]["name"] = label
    json_data[class_id]["color"] = []

    for c in range(3):
        value = random.randint(0, 255)
        json_data[class_id]["color"].append(value)

json_file = 'data/20-classes/classes.json'

# Write json file
with open(json_file, 'w') as fp:
    json.dump(json_data, fp, indent=4, sort_keys=True)
