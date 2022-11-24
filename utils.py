from dataclasses import dataclass
from numpy import array_equal

class dataReader:
    @staticmethod
    def read_data(dataset_id):
        #weights
        with open(f"./data/{dataset_id}_w.txt") as f:
            weights = [int(line) for line in f]
        #profits
        with open(f"./data/{dataset_id}_p.txt") as f:
            profits = [int(line) for line in f]
        #capacity
        with open(f"./data/{dataset_id}_c.txt") as f:
            capacity = int(f.readline())
        #opt. selection
        with open(f"./data/{dataset_id}_s.txt") as f:
            optimal_selection = [int(line) for line in f]

        return weights, profits, capacity, optimal_selection

@dataclass
class Item:
    id: int
    weight: int
    profit: int 

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)
