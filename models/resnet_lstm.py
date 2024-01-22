import torch 
import torch.nn as nn
import timm

class Resnet_BiLSTM(nn.Module):
    def __init__(
        self, 
        n_classes,
        vocab_length,
        img_model_name='resnet50',
        embed_dim=300,
        lstm_num_layers=2,
        lstm_hidden_size=128,
        dropout_prob=0.2):
        
        super(Resnet_BiLSTM, self).__init__()
        self.image_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=lstm_hidden_size)
        
        self.embedding = nn.Embedding(num_embeddings=vocab_length, embedding_dim=embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.layernorm = nn.LayerNorm(lstm_hidden_size*2) # bidirectional
        self.fc1 = nn.Linear(lstm_hidden_size*3, lstm_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(lstm_hidden_size, n_classes)
        
    def forward(self, img, text):
        
        img_features = self.image_encoder(img)
        
        # Convert 'text' tensor to LongTensor
        text = text.to(torch.int64)
        
        text_emb = self.embedding(text)
        
        lstm_out, _ = self.lstm(text_emb)
        
        lstm_out = lstm_out[:, -1, :]
        
        lstm_out = self.layernorm(lstm_out)
        combined = torch.cat((img_features, lstm_out), dim=1)
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    import torchinfo
    model = Resnet_BiLSTM(n_classes=2, vocab_length=1000)
    
        # Specify input sizes for both img and text
    img_input_size = (1, 3, 224, 224)  # Assuming input size of (batch_size, channels, height, width)
    text_input_size = (1, 30)  # Assuming input size of (batch_size, sequence_length)
    
    torchinfo.summary(model, input_size=[img_input_size, text_input_size])
        
        
