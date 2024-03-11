import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast

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

        
            
    def forward(self, pixel_values, input_ids):
        text_out = self.text_encoder(input_ids)
        image_out = self.visual_encoder(pixel_values)

        x = torch.cat((text_out, image_out), dim=1)
        x = self.classifier(x)
        
        return x

class CLIPVQA(nn.Module):
    def __init__(self, 
                model_id="openai/clip-vit-base-patch32",
                n_classes=2,
                classifier_type='lstm'):
        
        super(CLIPVQA, self).__init__()
        self.model = CLIPModel.from_pretrained(model_id)
        self.image_preprocessor = CLIPImageProcessor.from_pretrained(model_id)
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        
        self.vision_config = self.model.vision_model.config
        self.text_config = self.model.text_model.config
        # self.processor = CLIPProcessor.from_pretrained(model_id)

        self.fusion_dim = self.vision_config.projection_dim + self.text_config.projection_dim
        
        if classifier_type == 'linear':
            self.classifier = LinearClassifier(n_classes=n_classes, fusion_size=self.fusion_dim)

        elif classifier_type == 'lstm':
            self.classifier = LSTMClassifier(n_classes=n_classes, fusion_size=self.fusion_dim)
        
        print(f"freezing {model_id} parameters")
        for n, p in self.model.named_parameters():
            p.requires_grad = False
    
    def forward(self, pixel_values, input_ids):
        clip_output = self.model(input_ids, pixel_values)

        x = torch.cat(
            (clip_output.text_embeds, clip_output.image_embeds), 
            dim=1
            )
        # import pdb;pdb.set_trace()
        x = self.classifier(x)
    
        return x
    
if __name__ == '__main__':
    import torchinfo
    from PIL import Image
    import requests
     
    device = "cuda"
    # model = VQAModelFromPretrained('facebook/vit-mae-base')
    model = CLIPVQA()
    model = model.to(device)
    
    import pdb; pdb.set_trace()
    # config = model.text_encoder.config
    config = model.text_config
    
    input_image = torch.randn(size=(1, 3, 224, 224)).to(device)
    input_text_ids = torch.randint(0, config.vocab_size, size=(1, 77)).to(device)
    torchinfo.summary(model, input_data=[input_text_ids, input_image])
    
