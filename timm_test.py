import timm
import torch
import torchinfo 

# Choose a model architecture (e.g., 'resnet18')
model_name = 'resnet18'

# Load the pre-trained model
# model = timm.create_model(model_name, pretrained=True)
model = timm.create_model(model_name, pretrained=True, num_classes=128)

# Set the model to evaluation mode
model.eval()

# print(model)

# Example input (replace this with your own input)
example_input = torch.randn(1, 3, 224, 224)  # assuming input size of (batch_size, channels, height, width)

# Forward pass
with torch.no_grad():
    # output = model(example_input)
    torchinfo.summary(model, input_size=(1, 3, 224, 224))

# print("Model output shape:", output.shape)
