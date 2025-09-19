from os import walk, path
import json

filepath = 'data'
dataset = {
    "data": []
}

for root,_,files in walk(filepath):
    if len(files) != 0:
        for file in files:
            dataset['data'].append(path.join(root, file))

with open("dataset.json", 'w', encoding='utf-8') as file:
    json.dump(dataset, file)