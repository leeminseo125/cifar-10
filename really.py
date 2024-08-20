import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
mean = dataset.data.mean(axis=(0,1,2))
std = dataset.data.std(axis=(0,1,2))
mean = mean / 255
std = std / 255

print(mean, std)

class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.trainset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        image, label = self.trainset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean), (std))]) # 데이터 정규화
    
custom_dataset = MyDataset(transform=transform)

loader = DataLoader(dataset=custom_dataset, batch_size=8, shuffle=True)

data_iter = iter(loader)

images, labels = next(data_iter) # 이터

def denormalize(imgs, mean, std): #역정규화를 정의
    mean = torch.tensor(mean).view(1, 3, 1, 1) #view는 텐서의 차원에 따라 브로드 캐스팅이 불가능 할 수 있기에 모양을 바꿔줌.
    std = torch.tensor(std).view(1, 3, 1, 1)
    return imgs * std + mean

images = denormalize(images, mean, std).clamp(0, 1) #특정 픽셀의 값이 유효범위인 [0,1]을 벗어날 수 있기에 강제함

fig, axes = plt.subplots(2, 4, figsize=(10, 5)) # 각 사진 사이즈 설정

class_names = custom_dataset.trainset.classes # 데이터셋에 있는 클래스 지정
fig.canvas.manager.set_window_title('CIFAR-10') # 메인 타이틀 지정

for idx in range(8): # 8개의 사진 출력
    ax = axes[idx // 4, idx % 4] # 사진 위치 설정
    ax.imshow(np.transpose(images[idx].numpy(), (1, 2, 0))) # 넘파이 차원 설정
    ax.set_title(class_names[labels[idx]]) # 각 사진에 들어갈 타이틀
    
plt.show()
