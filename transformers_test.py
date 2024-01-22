# %%
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoTokenizer, RobertaModel, RobertaConfig
import torchinfo
from PIL import Image
import requests
import torch

## Image classification
# %%
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.to(device)
print(torchinfo.summary(model, input_size=(1,3,224,224)))

# %%
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image.save("test_VIT_image.png")

# %%
inputs = image_processor(images=image, return_tensors="pt").pixel_values
inputs = inputs.to(device)

# %%
outputs = model(inputs)

# %%
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


# text classification
# %%
text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Initializing a model (with random weights) from the configuration
model = Rmodel = RobertaModel.from_pretrained("roberta-base")

# %%
inputs = text_tokenizer("Hello, my dog is cute",
                        padding="max_length", 
                        max_length=20,
                        truncation=True,
                        return_tensors="pt")

# %%
outputs = model(inputs.input_ids)
last_hidden_states = outputs.last_hidden_state

