import numpy as np
import pickle
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


class Tokenizer():
    def __init__(self):
        self.values_to_num = {}
        self.time_to_num = {}
        self.instruments_to_num = {}

        self.num_to_values = {}
        self.num_to_times = {}
        self.num_to_instruments = {}

    def fit(self, all_data: np.ndarray):
        # change dtype to int
        all_data = all_data.astype(np.int64)
        values_count = 0
        time_count = 0
        instruments_count = 0

        values_array = np.unique(all_data[:, 0:7], axis=0)
        time_array = np.sort(np.unique(all_data[:, 7], axis=0))
        instruments_array = np.unique(all_data[:, 8:], axis=0)

        print('values starting')
        for i in tqdm(range(len(values_array))):
            self.values_to_num[str(values_array[i])] = values_count
            self.num_to_values[values_count] = values_array[i]
            values_count += 1
        print('values done')
        print('time starting')
        for i in tqdm(range(len(time_array))):
            self.time_to_num[str(time_array[i])] = time_count
            self.num_to_times[time_count] = time_array[i]
            time_count += 1
        print('time done')
        print('instruments starting')
        for i in tqdm(range(len(instruments_array))):
            self.instruments_to_num[str(instruments_array[i])] = instruments_count
            self.num_to_instruments[instruments_count] = instruments_array[i]
            instruments_count += 1
        print('instruments done')

    def tokenize(self, all_data: np.ndarray):
        # change dtype to int
        all_data = all_data.astype(np.int64)
        tokenized_data = np.zeros((all_data.shape[0], 3), dtype=np.int64)
        for i in tqdm(range(all_data.shape[0])):
            tokenized_data[i, 0] = self.values_to_num[str(all_data[i, 0:7])]
            tokenized_data[i, 1] = self.time_to_num[str(all_data[i, 7])]
            tokenized_data[i, 2] = self.instruments_to_num[str(all_data[i, 8:])]
        return tokenized_data

    def tokenize_all(self, all_data: list):
        """
        all_data: list of np.ndarrays
        returns: list of tokenized np.ndarrays
        """
        tokenized_data = []
        for i in tqdm(all_data):
            tokenized_data.append(self.tokenize(i))
        return tokenized_data

    def detokenize(self, tokenized_data: np.ndarray):
        # change dtype to int
        tokenized_data = tokenized_data.astype(np.int64)
        detokenized_data = np.zeros((tokenized_data.shape[0], 11), dtype=np.int64)
        for i in tqdm(range(tokenized_data.shape[0])):
            detokenized_data[i, 0:7] = self.num_to_values[tokenized_data[i, 0]]
            detokenized_data[i, 7] = self.num_to_times[tokenized_data[i, 1]]
            detokenized_data[i, 8:] = self.num_to_instruments[tokenized_data[i, 2]]
        return detokenized_data

    def detokenize_all(self, tokenized_data: list):
        detokenized_data = []
        for i in tqdm(tokenized_data):
            detokenized_data.append(self.detokenize(i))
        return detokenized_data

    def save_tokenizer(self, location='tokenizer.pickle'):
        with open(location, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('tokenizer saved at ' + location)

    def load_tokenizer(self, location='tokenizer.pickle'):
        with open(location, 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.values_to_num = tokenizer.values_to_num
        self.time_to_num = tokenizer.time_to_num
        self.instruments_to_num = tokenizer.instruments_to_num

        self.num_to_values = tokenizer.num_to_values
        self.num_to_times = tokenizer.num_to_times
        self.num_to_instruments = tokenizer.num_to_instruments
