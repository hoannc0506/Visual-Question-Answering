import lightning as L
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

def load_dataset(data_path, image_root="dataset/val2014-resised/"):
    data = []
    lines = open(data_path, 'r').read().strip().split('\n')
    for line in tqdm(lines, desc=f"Loading {data_path}"):
        temp = line.split('\t')
        qa = temp[1].split('?')
        
        answer = qa[2] if len(qa) == 3 else qa[1]

        data_sample = {
            'image_path': image_root+temp[0][:-2],
            'question': qa[0] + '?',
            'answer': answer.strip()
            }
        
        data.append(data_sample)

    return data


class VQADataModule(L.LightningDataModule):
    def __init__(self,
                 classes_to_idx,
                 image_preprocessor,
                 text_tokenizer,
                 ):
        super().__init__()
        self.classes_to_idx = classes_to_idx
        self.image_preprocessor = image_preprocessor
        self.text_tokenizer = text_tokenizer

    def custom_collate(self, batch):
        # import pdb; pdb.set_trace()
        # image_paths, questions, answers = zip(*batch)
        # images, questions, answers = [], [], []
        images = [Image.open(data['image_path']) for data in batch]
        questions = [data['question'] for data in batch]
        answers = [data['answer'] for data in batch]
        
        pixel_values = self.image_preprocessor(
            images=images, 
            return_tensors="pt"
            ).pixel_values
        
        input_ids = self.text_tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                ).input_ids
        
        targets = torch.tensor([self.classes_to_idx[label] for label in answers])
        
        return pixel_values, input_ids, targets

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.vqa_train = load_dataset('dataset/vaq2.0.TrainImages.txt')
            self.vqa_val = load_dataset('dataset/vaq2.0.DevImages.txt')

        if stage == "test":
            self.vqa_test = load_dataset('dataset/vaq2.0.TestImages.txt')

    def train_dataloader(self):
        return DataLoader(self.vqa_train, batch_size=64, shuffle=True, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.vqa_val, batch_size=32, shuffle=False, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return DataLoader(self.vqa_test, batch_size=32, shuffle=False, collate_fn=self.custom_collate)


if __name__ == "__main__":
    from transformers import CLIPImageProcessor, CLIPTokenizerFast
    
    model_id="openai/clip-vit-base-patch32"
    image_preprocessor = CLIPImageProcessor.from_pretrained(model_id)
    text_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    classes_to_idx = {'yes':1, 'no':0}
    
    vqa_dataset = VQADataModule(classes_to_idx, image_preprocessor, text_tokenizer)
    vqa_dataset.setup(stage='fit')
    
    train_loader = vqa_dataset.train_dataloader()
    import pdb; pdb.set_trace()
    pixel_values, input_ids, targets = next(iter(train_loader))
    
    print('here')