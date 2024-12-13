import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os
import csv
import gdown
import zipfile
import logging
import ast



root_dir = 'C:/Users/hashi/OneDrive/Documents/FYP-I/API/Attributes'
model_save_path = "attributes_trained.pth"

load_model = False
pretrained_path = "attributes_pre_trained.pth"

log_path = "attribute_log.txt"

# Hyperparameters
learning_rate=0.02
size_of_batch=64
num_of_workers=8
num_of_epochs=1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_file=os.path.join(root_dir,log_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])

def download_and_unzip(drive_url, output_zip_path, unzip_dir):
    if os.path.isdir(unzip_dir):
        logging.info("Dataset is already downloaded and unzipped.")
        return

    if not os.path.exists(output_zip_path):
        logging.info("Downloading dataset...")
        gdown.download(drive_url, output_zip_path, quiet=False)
    else:
        logging.info("Zip file already downloaded.")

    logging.info("Unzipping dataset...")
    with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    logging.info("Unzipping complete.")


drive_url = 'https://drive.google.com/uc?id=1qG5Xnxp8xJapIcE4xFLE6nEd8eHPilwK'
output_zip_path = os.path.join(root_dir, 'dataset.zip')
unzip_dir = os.path.join(root_dir, 'dataset')

download_and_unzip(drive_url, output_zip_path, unzip_dir)


def download_dataset(file_id, destination):
    if os.path.exists(destination):
        logging.info("File already downloaded.")
        return
    else:
        logging.info("Downloading File...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        logging.info("Download complete.")


file_id = '1oT-v72rrwfkcr3e920WF57dwPfnTwXWw'
train_path = os.path.join(root_dir, 'train.csv')
download_dataset(file_id, train_path)


# MODEL TRAINING
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained model normalization
])

class ClothingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor([int(val) for val in self.labels[idx]], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def read_valid_entries_from_csv(csv_file):
    img_paths = []
    img_labels = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                label = list(map(int, eval(row[1])))
                img_paths.append(os.path.join(root_dir, row[0]))
                img_labels.append(label)
            except Exception as e:
                print(f"Invalid entry in CSV: {row[0]}")
                print(e)               
    return img_paths, img_labels

image_paths , image_labels = read_valid_entries_from_csv(train_path)

train_dataset = ClothingDataset(image_paths, image_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=size_of_batch, shuffle=False, num_workers=num_of_workers)


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # 1000 output attributes

model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if load_model:
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
        epoch_loss = running_loss / len(train_loader)
        torch.save(model.state_dict(),model_save_path)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
    
    logging.info("Training complete.")
    torch.save(model.state_dict(),model_save_path)
    print("Model weights saved.")
    

train_model(model, criterion, optimizer, train_loader,num_epochs=num_of_epochs)
