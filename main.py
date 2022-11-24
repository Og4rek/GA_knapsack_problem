
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

if __name__ == '__main__':
    weights, profits, capacity, optimal_selection = dataReader.read_data("p01")
    print(weights, profits, capacity, optimal_selection)

        

