import numpy as np
import pickle
from tqdm import tqdm
from torch import tensor, Tensor
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

        self.token_values_count = []
        self.token_time_count = []
        self.token_instruments_count = []


    def fit(self, all_data):
        # change dtype to int
        if type(all_data) == tensor or type(all_data) == Tensor:
            all_data = all_data.numpy()

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

    def tokenize(self, all_data):
        # change dtype to int
        if type(all_data) == tensor or type(all_data) == Tensor:
            all_data = all_data.numpy()
        all_data = all_data.astype(np.int64)
        tokenized_data = np.zeros((all_data.shape[0], 3), dtype=np.int64)
        for i in tqdm(range(all_data.shape[0])):
            try:
                tokenized_data[i, 0] = self.values_to_num[str(all_data[i, 0:7])]
                tokenized_data[i, 1] = self.time_to_num[str(all_data[i, 7])]
                tokenized_data[i, 2] = self.instruments_to_num[str(all_data[i, 8:])]
            except KeyError:
                # Skip token if it's missing
                pass
        return tokenized_data

    def detokenize(self, tokenized_data):
        # change dtype to int
        if type(tokenized_data) == tensor or type(tokenized_data) == Tensor:
            tokenized_data = tokenized_data.numpy()
        tokenized_data = tokenized_data.astype(np.int64)
        detokenized_data = np.zeros((tokenized_data.shape[0], 11), dtype=np.int64)
        for i in tqdm(range(tokenized_data.shape[0])):
            try:
                detokenized_data[i, 0:7] = self.num_to_values[tokenized_data[i, 0]]
                detokenized_data[i, 7] = self.num_to_times[tokenized_data[i, 1]]
                detokenized_data[i, 8:] = self.num_to_instruments[tokenized_data[i, 2]]
            except KeyError:
                # Skip token if it's missing
                pass
        return detokenized_data

    def tokenize_all(self, all_data: list):
        """
        all_data: list of np.ndarrays
        returns: list of tokenized np.ndarrays
        """
        tokenized_data = []
        for i in tqdm(all_data):
            tokenized_data.append(self.tokenize(i))
        return tokenized_data

    def detokenize_all(self, tokenized_data: list):
        detokenized_data = []
        for i in tqdm(tokenized_data):
            detokenized_data.append(self.detokenize(i))
        return detokenized_data

    def fit_tokenize_clean(self, all_data, clean_amt):
        full_data = np.array([])
        for i in range(len(all_data)):
            full_data = np.append(full_data, all_data[i])
        full_data = full_data.reshape((-1,11))
        unique_data = np.unique(full_data, axis = 0)
        self.fit(unique_data)
        tokenized = self.tokenize_all(all_data)
        values_dict = {}
        time_dict = {}
        instruments_dict = {}
        for i in range(len(tokenized)):
            # add to index of token for each value/time/instrument
            for j in range(len(tokenized[i])):
                values_dict[tokenized[i][j][0]] = values_dict.get(tokenized[i][j][0], 0) + 1
                time_dict[tokenized[i][j][1]] = time_dict.get(tokenized[i][j][1], 0) + 1
                instruments_dict[tokenized[i][j][2]] = instruments_dict.get(tokenized[i][j][2], 0) + 1
        # if token is less than clean_amt, remove it from the dictionary along with tokenized data and tokenizer dicts
        # Remove tokens that appear fewer than clean_amt times
        for key, value in list(values_dict.items()):
            if value < clean_amt:
                del self.values_to_num[str(self.num_to_values[key])]
                del self.num_to_values[key]
        for key, value in list(time_dict.items()):
            if value < clean_amt:
                del self.time_to_num[str(self.num_to_times[key])]
                del self.num_to_times[key]
        for key, value in list(instruments_dict.items()):
            if value < clean_amt:
                del self.instruments_to_num[str(self.num_to_instruments[key])]
                del self.num_to_instruments[key]

        # Reindex the token dictionaries to remove gaps
        self._reindex_dicts()
        # Re-tokenize the data
        tokenized = self.tokenize_all(all_data)
        return tokenized

    def _reindex_dicts(self):
        # Reindex values_to_num
        new_values_to_num = {}
        new_num_to_values = {}
        index = 0
        for key in sorted(self.num_to_values.keys()):
            new_values_to_num[str(self.num_to_values[key])] = index
            new_num_to_values[index] = self.num_to_values[key]
            index += 1
        self.values_to_num = new_values_to_num
        self.num_to_values = new_num_to_values

        # Reindex time_to_num
        new_time_to_num = {}
        new_num_to_times = {}
        index = 0
        for key in sorted(self.num_to_times.keys()):
            new_time_to_num[str(self.num_to_times[key])] = index
            new_num_to_times[index] = self.num_to_times[key]
            index += 1
        self.time_to_num = new_time_to_num
        self.num_to_times = new_num_to_times

        # Reindex instruments_to_num
        new_instruments_to_num = {}
        new_num_to_instruments = {}
        index = 0
        for key in sorted(self.num_to_instruments.keys()):
            new_instruments_to_num[str(self.num_to_instruments[key])] = index
            new_num_to_instruments[index] = self.num_to_instruments[key]
            index += 1
        self.instruments_to_num = new_instruments_to_num
        self.num_to_instruments = new_num_to_instruments


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
