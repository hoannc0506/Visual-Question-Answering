# import torch
# import transformers
# from transformers import CLIPProcessor, CLIPModel
# # from vqa_model_from_pretrained import LSTMClassifier
# import torch.nn as nn
# import clip


        
   
'''     
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

path = "/workspace/CLIP/CLIP.png"
image = preprocess(Image.open(path)).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]    

''' 

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast


model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
image_preprocessor = CLIPImageProcessor.from_pretrained(model_id)
text_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_image = Image.open(requests.get(url, stream=True).raw)
input_prompt = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=["a photo of a cat", "a photo of a dog"], 
                   images=input_image, 
                   return_tensors="pt", 
                   padding='max_length'
                   )
# import pdb;pdb.set_trace()
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) 

print(probs)

input_ids = text_tokenizer(input_prompt,
                           return_tensors="pt", 
                           padding='max_length').input_ids

pixel_values = image_preprocessor(input_image, return_tensors="pt").pixel_values

import pdb;pdb.set_trace()
outputs = model(input_ids, pixel_values)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)
print(probs)