import torch 
import torch.nn as nn

class VisTrans_RoBERTa(nn.Module):
    def __init__(self):
        super(VisTrans_RoBERTa, self).__init__()
        pass
    
    def forward(self):
        pass
    
if __name__ == '__main__':
    import torchinfo
    model = VisTrans_RoBERTa(n_classes=2, vocab_length=1000)
    
        # Specify input sizes for both img and text
    img_input_size = (1, 3, 224, 224)  # Assuming input size of (batch_size, channels, height, width)
    text_input_size = (1, 30)  # Assuming input size of (batch_size, sequence_length)
    
    torchinfo.summary(model, input_size=[img_input_size, text_input_size])