import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

class VisualEncoder(nn.Module):
    def __init__(self, pretrained_name):
        super(VisualEncoder, self).__init__()
        self.model =  AutoModel.from_pretrained(pretrained_name)
        self.image_preprocessor = AutoImageProcessor.from_pretrained(pretrained_name)
        self.config = self.model.config

        print(f"freezing {pretrained_name} parameters")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
    
    def forward(self, x):
        outputs = self.model(x)

        return outputs.last_hidden_state[:, 0, :]
        
    
class TextEncoder(nn.Module):
    def __init__(self, pretrained_name):
        super(TextEncoder, self).__init__()
        self.model =  AutoModel.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.config = self.model.config

        print(f"freezing {pretrained_name} parameters")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
    def forward(self, x):
        outputs = self.model(x)
        
        return outputs.last_hidden_state[:, 0, :]

class LSTMClassifier(nn.Module):
    def __init__(self,
                 n_classes=2,
                 fusion_size=1536,
                 hidden_size=512,
                 n_layers=1,
                 dropout_prob=0.2,
                 ):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=fusion_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
            )
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # bidirectional x2 output hidden_size
        self.fc = nn.Linear(hidden_size*2, n_classes)
        
    def forward(self, x):
        x, _ = self.lstm(x) 
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class LinearClassifier(nn.Module):
    def __init__(self,
                 n_classes=2,
                 fusion_size=1536,
                 hidden_size=512,
                 dropout_prob=0.2,
                 ):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        
        return x


class VQAModelFromPretrained(nn.Module):
    def __init__(self,
                 visual_pretrained_name="google/vit-base-patch16-224",
                 text_pretrained_name="roberta-base",
                 classifier_type='lstm',
                 dropout_prob=0.2,
                 n_classes=2):
        super(VQAModelFromPretrained, self).__init__()
        
        self.visual_encoder = VisualEncoder(visual_pretrained_name)
        self.text_encoder = TextEncoder(text_pretrained_name)
        
        self.fusion_dim = self.visual_encoder.config.hidden_size + self.text_encoder.config.hidden_size

        if classifier_type == 'linear':
            self.classifier = LinearClassifier(n_classes=n_classes, fusion_size=self.fusion_dim)

        elif classifier_type == 'lstm':
            self.classifier = LSTMClassifier(n_classes=n_classes, fusion_size=self.fusion_dim)

            
    def forward(self, image, text):
        text_out = self.text_encoder(text)
        image_out = self.visual_encoder(image)

        x = torch.cat((text_out, image_out), dim=1)
        x = self.classifier(x)
        
        return x
    

if __name__ == '__main__':
    import torchinfo
    
    device = "cuda:3"
    model = VQAModelFromPretrained('facebook/vit-mae-base')
    model = model.to(device)
    
    # Specify input sizes for both img and text
    # img_input_size = (1, 3, 224, 224)  # Assuming input size of (batch_size, channels, height, width)
    # text_input_size = (1, 20) # Assuming input size of (batch_size, sequence_length)
    
    config = model.text_encoder.config
    input_image = torch.randn(size=(1, 3, 224, 224)).to(device)
    input_text_ids = torch.randint(0, config.vocab_size, size=(1, 300)).to(device)
    torchinfo.summary(model, input_data=[input_image, input_text_ids])