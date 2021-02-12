from Seq2seq.LSTM.utils.dataloader import *
from Seq2seq.LSTM.utils.model import *

import numpy as np

data_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train.csv"
train_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\train_set.csv"
valid_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\test_set.csv"
metadata_path = "E:\\MachineLearning\\Study\\RadiationReportSummarization\\Dataset\\char_level.txt"

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 80  # Number of samples to train on.

print("Define data loader and model")
lang = Lang(metadata_path)
train_data = Datagenerator(train_path, batch_size, True, lang, num_samples=num_samples)
valid_data = Datagenerator(valid_path, 1, False, lang)

num_encoder_tokens = lang.get_num_en_token()
num_decoder_tokens = lang.get_num_de_token()
input_characters = lang.get_in_char_set()
target_characters = lang.get_ta_char_set()
max_encoder_seq_length = lang.get_max_en_len()
max_decoder_seq_length = lang.get_max_de_len()

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Load model complete")

model.summary()

print("Training")
# Run training
from keras.optimizers import *

model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')
model.fit(x=train_data,
          epochs=epochs,
          validation_data=valid_path)

print("Training Complete")

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, lang.get_index('\t')] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = lang.get_reverse(sampled_token_index)
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = valid_data[seq_index]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', decode_sequence(input_seq))
    print('Decoded sentence:', decoded_sentence)
