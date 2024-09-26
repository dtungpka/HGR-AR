from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import time
Actions = ['Idle',
           'Pickup_item',
           'Use_item',
           'Aim',
           'Shoot'
           ]
def train(base_model):
    if base_model == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        #transfer learning to 5 classes
        model.fc = nn.Linear(model.fc.in_features, len(Actions))
    elif base_model == 'mobilenet_v2':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(Actions))
    elif base_model == 'vgg16':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(Actions))
    elif base_model == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, len(Actions))
    else:
        print('Invalid model specified')
        exit()

    #read dataset
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder('imgs_cropped_data/train', transform=data_transform)
    val_dataset = ImageFolder('imgs_cropped_data/val', transform=data_transform)
    test_dataset = ImageFolder('imgs_cropped_data/test', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model, loss function, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 20
    for epoch in tqdm(range(num_epochs),desc='Epoch'):
        model.train()
        disp_loss = 0
        for inputs, labels in tqdm(train_loader,desc=f"Epoch {epoch+1}/{num_epochs}: L={disp_loss:.4f}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            disp_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Test the model and calculate metrics
    model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    avg_times = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            st_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            avg_times.append(time.time()-st_time)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Results of {base_model} cropped:')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    print(classification_report(all_labels, all_preds))
    print(f'Avg Inference Time: {np.mean(avg_times):.4f} seconds')

    #save the model
    torch.save(model.state_dict(), f'{base_model}_cropped.pth')
if __name__ == '__main__':
    train('resnet50')
    train('mobilenet_v2')
    train('vgg16')
    train('densenet121')


