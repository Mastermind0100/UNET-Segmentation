import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from utils import OCTDataset, Compose, CropTransform, RandomHorizontalFlip, UNETAutoEncoder
from tqdm import tqdm

device = torch.device("cuda")
NUM_EPOCHS = 10

if __name__ == "__main__":
    print("Loading Model and Optimizer...")
    model = UNETAutoEncoder()
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.005, weight_decay=0.00000001, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = torch.nn.BCEWithLogitsLoss()
    global_step = 0

    print(f"Done. Model Loaded on {torch.cuda.get_device_name(0)}\n")
    
    print("Loading Dataset...")
    transforms = Compose([
        CropTransform(),
        RandomHorizontalFlip()
    ])

    dataset = OCTDataset(transform=transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    print("Done.\n")

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

        for images, rois in train_loader:
            images = images.to(device)
            rois = rois.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, rois)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss = loss.item())

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, rois in validation_loader:
                images = images.to(device)
                rois = rois.to(device)
                outputs = model(images)

                loss = criterion(outputs, rois)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(validation_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    torch.save(model.state_dict(), "model_state_dict.pth")