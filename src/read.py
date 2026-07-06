import numpy as np

def read_data(file_path):
    """ Read npy file and return the data as a numpy array. """

    data = np.load(file_path)
    print(f"Data loaded from {file_path}, shape: {data.shape}")
    print("type of data:", type(data))
    return data

input_data = "data/embeddings/100.dna.embeddings.npy"
data = read_data(input_data)