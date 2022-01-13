import pandas as pd
import numpy as np
import json


def store_dataset(dataloader):
    data_size = int(dataloader.dataset_X.shape[0])
    data_name = dataloader.name
    features = list(dataloader.dataset_X)
    data_info = json.dumps({'data_name': data_name, 'data_size': data_size, 'features': features},sort_keys=True,
                           indent=4, ensure_ascii=False)
    with open(str(data_name)+'.json', 'w', encoding='utf-8') as json_file:
        json.dump(data_info, json_file, ensure_ascii=False)
        print("write json file success!")