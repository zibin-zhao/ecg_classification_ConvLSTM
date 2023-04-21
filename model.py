"ECG model by Zibin ZHAO @HSING Group"

import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.5)

        pool_output_size = 3600 // 2 // 2 // 2

        self.lstm = nn.LSTM(512 * pool_output_size, 256, num_layers=3, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 3600)  # Reshape the input
        x = self.dropout1(self.pool1(self.relu1(self.batch_norm1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.batch_norm2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.batch_norm3(self.conv3(x)))))

        # Calculate the output size of the last pooling layer
        pool_output_size = x.size(2)

        x = x.contiguous().view(x.size(0), -1, 512 * pool_output_size)  # Flatten the tensor and keep the batch size

        _, (h_n, _) = self.lstm(x)
        x = self.dropout_lstm(h_n[-1])  # Apply dropout after the LSTM layer
        x = self.fc(x)
        x = self.softmax(x)
        return x

class ECGTransformer(nn.Module):
    def __init__(self):
        super(ECGTransformer, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.5)

        self.embedding_dim = 128    # match the conv output
        self.seq_len = 1800     # 3600 // 2

        self.pos_encoding = self.positional_encoding(self.embedding_dim, self.seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)

        self.fc = nn.Linear(self.embedding_dim * self.seq_len, 7)
        self.softmax = nn.Softmax(dim=1)

    def positional_encoding(self, d_model, seq_len):
        pos_encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0).to(device)

    def forward(self, x):
        x = x.view(-1, 1, 3600)  # Reshape the input
        x = self.dropout1(self.pool1(self.relu1(self.batch_norm1(self.conv1(x)))))

        x = x.permute(0, 2, 1)  # Transpose the tensor (batch_size, seq_len, embedding_dim)
        x += self.pos_encoding  # Add positional encoding

        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor and keep the batch size

        x = self.fc(x)
        x = self.softmax(x)
        return x