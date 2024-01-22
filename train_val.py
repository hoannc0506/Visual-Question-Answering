import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for images, questions, labels in tqdm(enumerate(dataloader)):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            outputs = model(images, questions)
            
            loss = criterion(outputs, labels)
            loss.append(loss.item())
            
            _, predictions = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
    loss = sum(losses) / len(losses)
    
    print(f"Validate loss: {loss:4f}")
    
    acc = correct / total
    
    return loss, acc


def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    model = model.train()
    epoch_train_losses = []
    
    for batch_idx, (images, questions, labels) in tqdm(enumerate(dataloader)):
        # forward
        images, questions, labels = images.to(device), questions.to(device), labels.to(device)
        outputs = model(images, questions)
        
        # compute loss
        train_loss = criterion(outputs, labels)

        # Backward and optimization
        optimizer.zero_grad()
        train_loss.backward()
        # update weights
        optimizer.step() 

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, loss: {train_loss.item():4f}")
        
        epoch_train_losses.append(train_loss.item())
        
    epoch_train_losses = sum(epoch_train_losses) / len(epoch_train_losses)
        
    return epoch_train_losses



def fit(model, train_loader, val_loade, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loade, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal accuracy: {val_acc:.4f}")
        
    return train_losses, val_losses
            