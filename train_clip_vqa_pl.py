import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import callbacks as pl_callbacks
import torch
from torch import optim, nn
from models.vqa_model_from_pretrained import CLIPVQA
from utils.vqa_pl_dataset import VQADataModule
from transformers import CLIPImageProcessor, CLIPTokenizerFast



# define the LightningModule
class LitCLIPVQA(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CLIPVQA()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        
        pixel_values, input_ids, labels = batch
        predictions = self.model(pixel_values, input_ids)

        loss = nn.functional.cross_entropy(predictions, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, labels = batch
        predictions = self.model(pixel_values, input_ids)

        val_loss = nn.functional.cross_entropy(predictions, labels)
        self.validation_step_outputs.append(val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val/loss', avg_val_loss)
        self.validation_step_outputs.clear()  # free memory
        
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


model_id="openai/clip-vit-base-patch32"
image_preprocessor = CLIPImageProcessor.from_pretrained(model_id)
text_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
classes_to_idx = {'yes':1, 'no':0}

wandb_logger = WandbLogger(project="VQA", name="clip-ViT-B-32")
vqa_model = LitCLIPVQA()
vqa_data_module = VQADataModule(classes_to_idx, image_preprocessor, text_tokenizer)

# Create a ModelCheckpoint callback to save the best and last checkpoints
checkpoint_callback = pl_callbacks.ModelCheckpoint(
    dirpath='./train_results/vqa_clip-vit-B32',
    filename='best',
    monitor='val/loss',
    mode='min',
    save_top_k=1,  # Save the best checkpoint
    save_last=True,  # Save the last checkpoint
)

# Create a Lightning Trainer with EarlyStopping
early_stop_callback = pl_callbacks.EarlyStopping(
    monitor='val/loss',
    patience=20,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'
)


# train model
trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback], # save every 1 epoch
    # limit_train_batches=10, 
    max_epochs=100,
    check_val_every_n_epoch=1,
    devices='auto',
    accelerator="gpu",
    enable_progress_bar=True,
    logger=wandb_logger,
    precision=16 # AMP
    )

# import pdb;pdb.set_trace()
trainer.fit(model=vqa_model, datamodule=vqa_data_module)