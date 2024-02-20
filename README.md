# Visual Question Answering project

## Dataset
[VQA COCO dataset](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view)

## Pipeline

![Pipeline](https://www.mdpi.com/applsci/applsci-13-05079/article_deploy/html/images/applsci-13-05079-g001.png)

### CNN + LSTM approach
+ Image Encoder: ResNet50
+ Text Encoder: BiLSTM


### Tranformers approach
+ Image Encoder: `Vision Transformer`, `ViTMAE`
+ Text Encoder: `RoBERTa base model`

## Train scripts
```

```

## Train results
+ [Wandb train results](https://wandb.ai/hoannc6/VQA)

| Models | Val acc | Test acc |
|----------|----------|----------|
| ResNet50 + LSTM | 0.5358 | - |
| VisTrans + RoBERTa (pooler_output) | 0.6690 | 0.6636 |
| [VisTrans + RoBERTa (last_hidden_state output)](https://drive.google.com/file/d/1hzeWyj58h8_lDyB-ysD684t_lcHfMQFB/view?usp=sharing) | **0.6931** | **0.6874** |

## To do
- [x] Public models
- [ ] Inference code
- [ ] Compare with other models