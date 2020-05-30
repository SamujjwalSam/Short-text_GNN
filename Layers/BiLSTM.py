# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_Classification
__classes__     : BiLSTM_Classifier
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""
import torch
import torch.nn as nn


class BiLSTM_Classifier(nn.Module):
    """ BiLSTM for classification.

    """

    # define all the layers used in model
    def __init__(self, vocab_size, hidden_dim, output_dim, embedding_dim=100,
                 n_layers=2, bidirectional=True, dropout=0.2, num_linear=1):
        super().__init__()  # Constructor

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout,
                            batch_first=True)

        self.linear_layers = []
        for _ in range(num_linear - 1):
            if bidirectional:
                self.linear_layers.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            else:
                self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.linear_layers = nn.ModuleList(self.linear_layers)

        # Final dense layer
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

        # activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        """ Takes ids of input text, pads them and predict using BiLSTM.

        Args:
            text:
            text_lengths:

        Returns:

        """
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
        #                                                     text_lengths,
        #                                                     batch_first=True)

        # packed_output1, (hidden1, cell) = self.lstm(packed_embedded)
        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
        # cell = [batch size, num num_lstm_layers * num directions, hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        for layer in self.linear_layers:
            hidden = layer(hidden)

        # hidden = [batch size, hid dim * num directions]
        logits = self.fc(hidden)

        # Final activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # logits = self.act(logits)

        return logits


def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    pass


if __name__ == "__main__":
    main()
