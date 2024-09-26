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
from PIL import Image
import os
import pandas as pd
import time

HEIGHT = 480
WIDTH = 640
Actions = ['Idle',
           'Pickup_item',
           'Use_item',
           'Aim',
           'Shoot'
           ]




data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#load hand_posture_model.pth
class HandPostureDataset(Dataset):
    def __init__(self, dataframe,imgs_path,flip_hand=True,angle_hand=False):
        self.data = dataframe['landmarks'].values
        self.handness = dataframe['handness'].values
        self.labels = dataframe['labels'].values  
        self.frame_ids = dataframe['frame_ids'].values
        self.fid = dataframe['fid'].values
        self.flip_hand = flip_hand
        self.angle_hand = angle_hand
        self.imgs_path = imgs_path

        self.last_sample = None
        self.last_img = None
        self.last_label = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.feature_extraction(sample,self.handness[idx])
        #normalize to 0-1
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        #print(sample.shape)
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        sample_frame_name = os.path.join(self.imgs_path,Actions[self.labels[idx]],f"{self.fid[idx]}_{self.frame_ids[idx]}.jpg")
        #if sample not exist, return last sample
        if not os.path.exists(sample_frame_name):
            print(f"Image {sample_frame_name} not found")
            return self.last_sample, self.last_img, self.last_label
        
        img = Image.open(sample_frame_name)
        img = data_transform(img)

        self.last_sample = sample
        self.last_label = label
        self.last_img = img

        return sample, img, label
        

    def feature_extraction(self,landmarks,handness):
        PARSE_LANDMARKS_JOINTS = [    
        [0, 1], [1, 2], [2, 3], [3, 4], # thumb
        [0, 5],[5, 6], [6, 7], [7, 8], # index finger
        [5, 9],[9,10],[10, 11], [11, 12],# middle finger
        [9, 13],[13, 14],[14, 15],[15, 16], # ring finger
        [13, 17],  [17, 18], [18, 19], [19,20]   # little finger
        ]   
        def calculate_angle(landmark1, landmark2):
            return np.math.atan2(np.linalg.det([landmark1, landmark2]), np.dot(landmark1, landmark2))
        if self.flip_hand and handness == 1:
            landmarks = landmarks * np.array([-1,1])
        if self.angle_hand:
            angles = []
            for joint in PARSE_LANDMARKS_JOINTS:
                angle = calculate_angle(landmarks[joint[0]],landmarks[joint[1]])
                angles.append(angle)
            return np.array([angles,angles]).T
        return landmarks
            


# Model definition
class HandPostureModel(nn.Module):
    def __init__(self):
        super(HandPostureModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.lstm = nn.LSTM(input_size=128, hidden_size=100, batch_first=True)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, len(Actions))  

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)
def fusion_model(base_model):
    model_hand = HandPostureModel()

    model_hand.load_state_dict(torch.load('hand_posture_model.pth'))

    #to fit with late fusion withh image, change fc2  from 5 to 1000
    model_hand.fc2 = nn.Linear(50, 1000)

    if base_model == 'resnet50':
        model_image = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        #transfer learning to 5 classes
        model_image.fc = nn.Linear(model_image.fc.in_features, len(Actions))
        #load 'resnet50.pth'
        model_image.load_state_dict(torch.load('resnet50.pth'))

        model_image.fc = nn.Linear(model_image.fc.in_features, 1000)
    elif base_model == 'mobilenet_v2':
        model_image = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model_image.classifier[1] = nn.Linear(model_image.classifier[1].in_features, len(Actions))
        #load 'mobilenet_v2.pth'
        model_image.load_state_dict(torch.load('mobilenet_v2.pth'))
        model_image.classifier[1] = nn.Linear(model_image.classifier[1].in_features, 1000)
    elif base_model == 'vgg16':
        model_image = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        model_image.classifier[6] = nn.Linear(model_image.classifier[6].in_features, len(Actions))
        #load 'vgg16.pth'
        model_image.load_state_dict(torch.load('vgg16.pth'))
        model_image.classifier[6] = nn.Linear(model_image.classifier[6].in_features, 1000)
    elif base_model == 'densenet121':
        model_image = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        model_image.classifier = nn.Linear(model_image.classifier.in_features, len(Actions))
        #load 'densenet121.pth'
        model_image.load_state_dict(torch.load('densenet121.pth'))
        model_image.classifier = nn.Linear(model_image.classifier.in_features, 1000)
    else:
        print('Invalid model specified')
        exit()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    df = pd.read_hdf('skeleton.h5')


    # Manually set test and validation IDs
    test_ids = ['1584', '0246', '0110', '0013']
    val_ids = ['2089', '1596']

    # Get unique IDs
    unique_fids = df['fid'].unique()

    # Determine training IDs by excluding test and validation IDs
    train_ids = [fid for fid in unique_fids if fid not in test_ids + val_ids]

    # Create dataframes for each set
    train_df = df[df['fid'].isin(train_ids)]
    val_df = df[df['fid'].isin(val_ids)]
    test_df = df[df['fid'].isin(test_ids)]

    # Verify the splits
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")


    train_dataset = HandPostureDataset(train_df,'imgs_data/train/')
    val_dataset = HandPostureDataset(val_df,'imgs_data/val/')
    test_dataset = HandPostureDataset(test_df,'imgs_data/test/')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #init a model with late fusion
    class LateFusionModel(nn.Module):
        def __init__(self, model_hand, model_image):
            super(LateFusionModel, self).__init__()
            self.model_hand = model_hand
            self.model_image = model_image
            self.fc = nn.Linear(2000, len(Actions))

        def forward(self, x_hand, x_image):
            x_hand = self.model_hand(x_hand)
            x_image = self.model_image(x_image)
            x = torch.cat((x_hand, x_image), dim=1)
            x = self.fc(x)
            return torch.softmax(x, dim=1)
        
    model = LateFusionModel(model_hand, model_image).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs_hand, inputs_image, labels in tqdm(train_loader):
            inputs_hand, inputs_image, labels = inputs_hand.to(device), inputs_image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_hand, inputs_image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Step the scheduler
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs_hand, inputs_image, labels in val_loader:
                inputs_hand, inputs_image, labels = inputs_hand.to(device), inputs_image.to(device), labels.to(device)
                outputs = model(inputs_hand, inputs_image)
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
        for inputs_hand, inputs_image, labels in test_loader:
            start_time = time.time()
            inputs_hand, inputs_image, labels = inputs_hand.to(device), inputs_image.to(device), labels.to(device)
            outputs = model(inputs_hand, inputs_image)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            avg_times.append(time.time() - start_time)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Late Fusion Model with {base_model} backbone:")

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    print(classification_report(all_labels, all_preds))
    print(f'Average inference time: {np.mean(avg_times):.4f} seconds')

    torch.save(model.state_dict(), f'late_fusion_{base_model}.pth')

if __name__ == '__main__':
    fusion_model('mobilenet_v2')
    fusion_model('resnet50')
    fusion_model('densenet121')
    fusion_model('vgg16')
    
