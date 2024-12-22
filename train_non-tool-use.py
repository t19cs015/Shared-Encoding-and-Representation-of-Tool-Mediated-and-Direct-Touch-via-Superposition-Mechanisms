import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(42)

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class MyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data_list = self.load_data()

    def load_data(self):
        data_list = []
        for file_path in self.file_paths:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)
        return data_list

    def __len__(self):
        return sum(len(data) for data in self.data_list)

    def __getitem__(self, idx):
        for data in self.data_list:
            if idx < len(data):
                return data[idx]
            idx -= len(data)
        raise IndexError("Index out of range")

class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VisionDecoder(nn.Module):
    def __init__(self):
        super(VisionDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128 * 4 * 6),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 6)
        x = self.decoder(x)
        return x

class MotionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TactileEncoder(nn.Module):
    def __init__(self):
        super(TactileEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class TactileDecoder(nn.Module):
    def __init__(self):
        super(TactileDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32*4*4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 4, 4)
        x = self.decoder(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisionEncoder()
        self.decoder = VisionDecoder()
        self.lstm = nn.LSTM( 
            input_size=128+3+64, # input_size=128+3+64,
            hidden_size=128, # 128
            num_layers=2,
            batch_first=True
        )
        self.motionencoder = MotionEncoder(3, 5, 3)
        self.tactileencoder = TactileEncoder()
        self.tactiledecoder = TactileDecoder()

    def gen_mask(self, B, T, p):
        p_mask = p
        mask = torch.rand(B, T)
        #mask[:, 0] = 1
        mask[:, :20] = 1
        mask = mask > p_mask
        mask = mask.float()
        mask = mask.view(B*T, 1)
        return mask

    def forward(self, v, m, t):
        vB, vT, vch, vH, vW = v.shape


        v = v.view(vB*vT, vch, vH, vW)
        venc = self.encoder(v)
        
        mask = self.gen_mask(vB, vT, 0.0).to(v.device) # vision
        venc = mask*venc
        venc = venc.view(vB, vT, -1)

        B, T, X = m.shape
        m = m.view(B*T, X)
        motion = self.motionencoder(m)
        motion = motion.view(B, T, -1)

        B, T, ch, H, W = t.shape
        t = t.view(B*T, ch, H, W)
        tactile = self.tactileencoder(t)
        
        mask = self.gen_mask(B, T, 0.0).to(t.device) # touch
        tactile = mask*tactile
        tactile = tactile.view(B, T, -1)

        enc = torch.cat([venc, motion, tactile], -1) #enc = torch.cat([venc, motion, tactile], -1)
        h, _ = self.lstm(enc)
        hh = h.reshape(vB*vT, -1)
        
        vdec = self.decoder(hh)
        vdec = vdec.view(vB, vT, vch, vH, vW)

        tdec = self.tactiledecoder(hh)
        tdec = tdec.view(vB, vT, 1, 16, 16)

        return vdec, h, mask.view(vB, vT), tdec, mask.view(B, T)

# データセットとデータローダーの作成
file_paths = [
    "train-data_non-tool-use.pkl",
]

combined_train_dataset = MyDataset(file_paths)

with open("test-data_non-tool-use.pkl", 'rb') as file:
    ds_test = pickle.load(file)

# with open("random/test.pkl", 'rb') as file:
train_loader = DataLoader(combined_train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=4, shuffle=True)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 400
train_loss = []
test_loss = []
train_vision_loss = []
train_touch_loss = []
test_vision_loss = []
test_touch_loss = []


for epoch in range(num_epochs):
    if epoch % 100 == 0:
        torch.cuda.empty_cache()

    model.train()
    epoch_train_loss = 0
    epoch_train_vision_loss = 0
    epoch_train_touch_loss = 0
    for data in train_loader:
        inputs, motions, tactiles, targets, tactileTargets = [d.to(device) for d in data]
        optimizer.zero_grad()
        outputs = model(inputs, motions.float(), tactiles)
        loss1 = criterion(outputs[3], tactileTargets) # touch
        loss2 = criterion(outputs[0], targets) # vision
        loss = loss2 #  loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_vision_loss += loss2.item()
        epoch_train_touch_loss += loss1.item()


    model.eval()
    epoch_test_loss = 0
    epoch_test_vision_loss = 0
    epoch_test_touch_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, motions, tactiles, targets, tactileTargets = [d.to(device) for d in data]
            outputs = model(inputs, motions.float(), tactiles.float())
            loss1 = criterion(outputs[3], tactileTargets)
            loss2 = criterion(outputs[0], targets)
            loss = loss2
            epoch_test_loss += loss.item()
            epoch_test_vision_loss += loss2.item()
            epoch_test_touch_loss += loss1.item()



    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_test_loss = epoch_test_loss / len(test_loader)
    avg_train_vision_loss = epoch_train_vision_loss / len(train_loader)
    avg_test_vision_loss = epoch_test_vision_loss / len(test_loader)
    avg_train_touch_loss = epoch_train_touch_loss / len(train_loader)
    avg_test_touch_loss = epoch_test_touch_loss / len(test_loader)
    
    train_loss.append(avg_train_loss)
    test_loss.append(avg_test_loss)
    train_vision_loss.append(avg_train_vision_loss)
    test_vision_loss.append(avg_test_vision_loss)
    train_touch_loss.append(avg_train_touch_loss)
    test_touch_loss.append(avg_test_touch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"result/model4_epoch_{epoch+1}.pth")

# 最終モデルの保存
torch.save(model.state_dict(), "result/model4.pth")

# ロスの保存
with open("result/model4-trainloss.pkl", 'wb') as f:
    pickle.dump(train_loss, f)

with open("result/model4-testloss.pkl", 'wb') as f:
    pickle.dump(test_loss, f)


with open("result/model4-trainloss-vision.pkl", 'wb') as f:
    pickle.dump(train_vision_loss, f)

with open("result/model4-testloss-vision.pkl", 'wb') as f:
    pickle.dump(test_vision_loss, f)


with open("result/model4-trainloss-touch.pkl", 'wb') as f:
    pickle.dump(train_touch_loss, f)

with open("result/model4-testloss-touch.pkl", 'wb') as f:
    pickle.dump(test_touch_loss, f)

# ロスのプロット
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='train')
plt.plot(range(1, len(test_loss) + 1), test_loss, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.ylim(0, 0.02)
plt.legend()

plt.grid(True)
plt.savefig('result/model4-loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Final train loss: {train_loss[-1]:.4f}, Final test loss: {test_loss[-1]:.4f}")