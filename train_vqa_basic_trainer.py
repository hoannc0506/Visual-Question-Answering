import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.vqa_dataset_from_pretrained import VQADatasetFromPretrained, load_dataset
from models.vqa_model_from_pretrained import VQAModelFromPretrained

from utils import trainer
import wandb
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some data with a given model')
    parser.add_argument('--visual-pretrained', default='google/vit-base-patch16-224', help='Name of images pretrained')
    parser.add_argument('--text-pretrained', default='roberta-base', help='Name of text pretrained')
    parser.add_argument('--device', default="cuda:3")

    args = parser.parse_args()

    visual_pretrained = args.visual_pretrained
    text_pretrained = args.text_pretrained
    device = args.device
    
    # load data
    train_data = load_dataset('dataset/vaq2.0.TrainImages.txt')
    val_data = load_dataset('dataset/vaq2.0.DevImages.txt')
    test_data = load_dataset('dataset/vaq2.0.TestImages.txt')
    
    print(len(train_data), len(val_data), len(test_data))
    
    # create mapping dict
    classes = set([sample['answer'] for sample in train_data])
    classes_to_idx = {cls_name:idx for idx, cls_name in enumerate(classes)}
    idx_to_classes = {idx:cls_name for idx, cls_name in enumerate(classes)}
    
    # create model, tokenizer, preprocessor
    model = VQAModelFromPretrained(
        visual_pretrained_name=visual_pretrained,
        text_pretrained_name=text_pretrained,
        n_classes=len(classes)
    )
    text_tokenizer = model.text_encoder.tokenizer
    image_preprocessor = model.visual_encoder.image_preprocessor
    model = model.to(device)
    
    # create datasets
    train_dataset = VQADatasetFromPretrained(train_data, classes_to_idx, image_preprocessor, text_tokenizer)
    val_dataset = VQADatasetFromPretrained(val_data, classes_to_idx, image_preprocessor, text_tokenizer)
    test_dataset = VQADatasetFromPretrained(test_data, classes_to_idx, image_preprocessor, text_tokenizer)
    
    # create dataloaders
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 32
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    # sample
    # image, question, label = next(iter(train_loader))
    # print(image.shape, question.shape, label.shape)

    #
    lr = 1e-2
    epochs = 50
    weight_decay = 1e-5
    scheduler_step_size = epochs *0.6
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)
    
    # start a new wandb run to track this script
    wandb_logger = wandb.init(
        project="VQA",
        name=f"{visual_pretrained.replace('/', '-')}_{text_pretrained.replace('/', '-')}"
    )
    
    # train model
    train_losses, val_losses = trainer.fit(model, train_loader, val_loader, criterion, 
                                             optimizer, scheduler, device, epochs, logger=wandb_logger)
    
    val_loss, val_acc = trainer.evaluate(model, val_loader, criterion, device)
    
    test_loss, test_acc = trainer.evaluate(model, test_loader, criterion, device)
    
    print("Val acc", val_acc)
    print("Test acc", test_acc)

if __name__ == "__main__":
    main()