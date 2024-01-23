import torch 
import torch.nn as nn
from transformers import ViTModel
from transformers import RobertaModel

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.model =  ViTModel.from_pretrained("google/vit-base-patch16-224")
        
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs.pooler_output
        
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model =  RobertaModel.from_pretrained("roberta-base")
    
    def forward(self, x):
        outputs = self.model(x)
        
        return outputs.pooler_output


class Classifier(nn.Module):
    def __init__(self, 
                 hidden_size=512,
                 n_layers=1,
                 dropout_prob=0.2,
                 n_classes=2):
        super(Classifier, self).__init__()
        
        # concat 2 visual and textual encoder outputs
        lstm_imput_size = 768*2
        
        self.lstm = nn.LSTM(
            input_size=lstm_imput_size,
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


class VisTrans_RoBERTa(nn.Module):
    def __init__(self, 
                 hidden_size=512,
                 n_layers=1,
                 dropout_prob=0.2,
                 n_classes=2):
        super(VisTrans_RoBERTa, self).__init__()
        self.visual_encoder = VisualEncoder()
        self.text_encoder = TextEncoder()
        
        self.classifier = Classifier(hidden_size,
                                     n_layers,
                                     dropout_prob,
                                     n_classes)
    
    def forward(self, image, text):
        text_out = self.text_encoder(text)
        image_out = self.visual_encoder(image)
        x = torch.cat((text_out, image_out), dim=1)
        x = self.classifier(x)
        
        return x
    
    def freeze(self):
        # freeze image and text encoders pretrained
        for n, p in self.visual_encoder.named_parameters():
            p.requires_grad = False
        
        for n, p in self.text_encoder.named_parameters():
            p.requires_grad = False
    
    
if __name__ == '__main__':
    import torchinfo
    n_classes = 2
    hidden_size = 1024
    n_layers = 1
    device = "cuda:2"

    model = VisTrans_RoBERTa(hidden_size=hidden_size,
                            n_layers=n_layers,
                            n_classes=n_classes)

    model = model.to(device)
    model.freeze()
    
    # Specify input sizes for both img and text
    img_input_size = (1, 3, 224, 224)  # Assuming input size of (batch_size, channels, height, width)
    text_input_size = (1, 20) # Assuming input size of (batch_size, sequence_length)
    
    torchinfo.summary(model, input_size=[img_input_size, text_input_size])