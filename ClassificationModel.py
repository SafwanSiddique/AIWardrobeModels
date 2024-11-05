import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import csv
import gdown
import zipfile
import logging

# Set up logging
log_file_path = os.path.join('/Users/apple/Documents/AiWardrobe/cv part/cvPart', 'classification_training_log.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

root_dir = '/Users/apple/Documents/AiWardrobe/cv part/cvPart'
model_save_path = "classification_trained.pth"
pretrained_path = "classification_pre_trained.pth"
load_model = False

# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 1
num_of_workers = 4


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


file_id = '1Vjaq1ExMr4h0U99esaTzoxdUC7x7QiuL'
train_path = os.path.join(root_dir, 'train.csv')

download_dataset(file_id, train_path)

test_path = os.path.join(root_dir, 'test.csv')
file_idt = '1-1iN7bvo0rAhLf9xkVa4O62oBTatmRqt'
download_dataset(file_idt, test_path)

# MODEL TRAINING

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")  # Open the image
        except FileNotFoundError:
            logging.info(f"Image not found at path: {img_path}")
            return None, label

        if self.transform:
            image = self.transform(image)

        return image, label


def read_valid_entries_from_csv(csv_file):
    img_paths = []
    img_labels = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                img_paths.append(os.path.join("dataset", row[0]).replace("\\", "/"))
                img_labels.append(int(row[1]))
            except Exception:
                logging.info(f"Invalid entry in CSV: {row}")

    return img_paths, img_labels


train_paths, train_labels = read_valid_entries_from_csv(train_path)

dataset = CustomImageDataset(train_paths, train_labels, transform=transform)
data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_of_workers)

# Initialize model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 50)

# Loading the trained weights
if load_model:
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

# Move model to the appropriate device (GPU/CPU)
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(model, data_loader, criterion, optimizer, num_epochs):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
    logging.info("Training complete.")


train_model(model, data_loader, criterion, optimizer, num_epochs=num_epochs)
torch.save(model.state_dict(), model_save_path)
