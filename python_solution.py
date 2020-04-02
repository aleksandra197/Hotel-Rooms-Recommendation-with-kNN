# Task 4

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Data load and transformation

data = pd.read_pickle(r'jr_data_analyst.pickle')

data.replace('', np.nan, inplace=True)
data = data.dropna()
data.reset_index(drop=True, inplace=True)
data.index.names = ['queryId']

# Unnesting room IDs

data = data.explode('roomIdsReturned')

# Creating binary matrix of room appearance in queries

data = data[['roomIdsReturned']]
data.reset_index(level=0, inplace=True)

data["value"] = 1
df = pd.pivot_table(data, values="value", index=["roomIdsReturned"], columns="queryId", fill_value=0)

rooms = pd.DataFrame({'roomId': data['roomIdsReturned'].unique()}).sort_values(by=['roomId'])
rooms = rooms.reset_index(drop=True)

# Applying KNN to find similar vectors

nbrs = NearestNeighbors(algorithm='ball_tree', n_neighbors=11).fit(df)
distances, indices = nbrs.kneighbors(df)

# Scoring

def print_similar_rooms(id):
    index = rooms[rooms['roomId'] == id].index.values[0]
    array = indices[index][1:]
    results = pd.DataFrame({'Similar_rooms_IDs': rooms['roomId'][array[0:]]})
    results = results.reset_index(drop=True)
    results.index += 1
    return results


results1 = print_similar_rooms(157)
results2 = print_similar_rooms(198)
results3 = print_similar_rooms(377)

# Saving to excel file

writer = pd.ExcelWriter('results_task4.xlsx')

results1.to_excel(writer, sheet_name='room_157')
results2.to_excel(writer, sheet_name='room_198')
results3.to_excel(writer, sheet_name='room_377')

writer.save()
