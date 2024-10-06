import torch.nn as nn
import torch.nn.functional as F

class LSTM_Model(nn.Module): #network architecture
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, lstm_dropout,bidirectional):
        super(LSTM_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, vocab_size)
    
    def forward(self, inputs):
        x = self.embeddings(inputs)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x