# Visual Question Answering project

## Dataset
[VQA COCO dataset](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view)

## Pipeline

![Pipeline](https://www.researchgate.net/publication/328128164/figure/fig1/AS:678896051687424@1538872840192/Visual-Question-Answering-Common-Approach-Fig1-indicates-the-steps-of-VQA-approach-at.png)

### CNN + LSTM approach
+ Image feature extractor: ResNet50
+ Text tokenizer: BiLSTM


### Tranformers approach
+ Image feature extractor: `Vision Transformer`
+ Text tokenizer: `RoBERTa base model`
+ Experiments with 2 output layers: `outputs.pooler_output` and `outputs.last_hidden_state[:, 0, :]`

## Train results
+ [Wandb train results](https://wandb.ai/hoannc6/VQA)

| Models | Val acc | Test acc |
|----------|----------|----------|
| ResNet50 + LSTM | 0.5358 | - |
| VisTrans + RoBERTa (pooler_output) | 0.6690 | 0.6636 |
| VisTrans + RoBERTa (last_hidden_state output) | **0.6931** | **0.6874** |

## To do
- [ ] Public models
- [ ] Inference code
- [ ] Compare with other models