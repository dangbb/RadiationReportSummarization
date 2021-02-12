import json

import numpy as np
import pandas as pd

from keras.utils import *

data_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train.csv"
train_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train_set.csv"
valid_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\test_set.csv"
metadata_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\char_level.txt"

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100  # Number of samples to train on.


class Lang:
    def __init__(self, metadata_path):
        self.path = metadata_path
        self.load()

    def load(self):
        with open(self.path) as file:
            data = json.load(file)
            self.metadata = data['metadata'][0]
            self.input_token_dict = dict([(x, i) for i, x in enumerate(self.metadata['input_char'])])
            self.output_token_dict = dict([(x, i) for i, x in enumerate(self.metadata['target_char'])])
            self.reverse_input_token_dict = dict([(i, x) for i, x in enumerate(self.metadata['input_char'])])
            self.reverse_output_token_dict = dict([(i, x) for i, x in enumerate(self.metadata['target_char'])])

    def get_index(self, char):
        return self.input_token_dict[char]

    def get_reverse(self, index):
        return self.reverse_input_token_dict[index]

    def get_in_char_set(self):
        return self.metadata['input_char']

    def get_ta_char_set(self):
        return self.metadata['target_char']

    def get_num_en_token(self):
        return self.metadata['num_en_token']

    def get_num_de_token(self):
        return self.metadata['num_de_token']

    def get_max_en_len(self):
        return self.metadata['max_en_seq_len']

    def get_max_de_len(self):
        return self.metadata['max_de_seq_len']

    def encode(self, input):
        """

        :param input: a array of strings, form (inputs, outputs), where inputs/outputs is a array of string
        :return: 3 onehot vector
        """
        encoder_input = np.zeros((len(input[0]), self.get_max_en_len(), len(self.get_in_char_set())), dtype='float32')
        decoder_input = np.zeros((len(input[1]), self.get_max_de_len(), len(self.get_ta_char_set())), dtype='float32')
        decoder_target = np.zeros((len(input[1]), self.get_max_de_len(), len(self.get_ta_char_set())), dtype='float32')

        for i, (input, output) in enumerate(zip(input[0], input[1])):
            output = '\t' + output + '\n'
            for t, s in enumerate(input):
                encoder_input[i, t, self.get_index(s)] = 1
            for t, s in enumerate(output):
                decoder_input[i, t, self.get_index(s)] = 1
                if t > 0:
                    decoder_target[i, t - 1, self.get_index(s)] = 1
        return ([encoder_input, decoder_input], decoder_target)

    def decode(self, input):
        """

        :param input: 3D array, onehot, size (num, len, max_char)
        :return: 1 string array
        """
        res = []
        for i in range(len(input)):
            ans = ""
            for j in range(len(input[i])):
                for k in range(len(input[i][j])):
                    if input[i][j][k] == 1:
                        ans = ans + self.get_reverse(k)
                        break
            res.append(ans)
        return res


class Datagenerator(Sequence):
    def __init__(self, path, batch_size, shuffle, lang: Lang, num_samples=None):
        self.train_path = path
        self.lang = lang
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_samples = num_samples

        df = pd.read_csv(path)
        self.total = len(df)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        else:
            return int(np.ceil(self.total / self.batch_size))

    def __getitem__(self, idx):
        """

        :param idx:
        :return: haven't decode
        """
        if idx < self.__len__():
            df = pd.read_csv(self.train_path, skiprows=(idx-1)*self.batch_size, nrows=self.batch_size, header=None, index_col=None)
            if self.shuffle:
                df.sample(frac=1)
            return self.lang.encode((df[3], df[4]))
        else:
            df = pd.read_csv(self.train_path, skiprows=(idx-1)*self.batch_size, header=None, index_col=None)
            if self.shuffle:
                df.sample(frac=1)
            return self.lang.encode((df[3], df[4]))


class DataLoader:
    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path

    def load_data(self, is_train=True):
        path = self.train_path if is_train else self.valid_path
        df = pd.read_csv(path, index_col=0, header=None)
        inputs = [x for x in df[3]]
        outputs = [x for x in df[4]]
        data = (inputs, outputs)
        return data

    def generate_data(self, batch_size, step, lang: Lang, is_train=True):
        path = self.train_path if is_train else self.valid_path
        idx = 1

        while True:
            df = pd.read_csv(path, skiprows=(idx - 1) * batch_size, nrows=batch_size, header=None, index_col=0)
            inputs = [x for x in df[3]]
            outputs = []
            for x in df[4]:
                outputs.append('\t' + x + '\n')
            data = (inputs, outputs)
            yield lang.encode(data)

            if idx < step:
                idx = idx + 1
            else:
                break

"""lang = Lang(metadata_path)
dataloader = DataLoader(train_path, valid_path)

gen = dataloader.generate_data(batch_size, 10, lang, is_train=False)

for x in gen:
    print(x)
    break
"""

