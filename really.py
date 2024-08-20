import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 데이터 정규화

    def __call__(self, image):
        return self.transform(image)

class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.trainset = datasets.CIFAR10(root='./data', train=True, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        image, label = self.trainset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# CustomTransform 클래스를 사용하여 transform을 정의
custom_transform = Transform()
custom_dataset = MyDataset(transform=custom_transform)

loader = DataLoader(dataset=custom_dataset, batch_size=8, shuffle=True)

data_iter = iter(loader)

images, labels = next(data_iter) # 이터

images = images / 2 + 0.5  # 데이터 표준화 복원

fig, axes = plt.subplots(2, 4, figsize=(10, 5)) # 각 사진 사이즈 설정

class_names = custom_dataset.trainset.classes # 데이터셋에 있는 클래스 지정
fig.canvas.manager.set_window_title('CIFAR-10') # 메인 타이틀 지정

for idx in range(8): # 8개의 사진 출력
    ax = axes[idx // 4, idx % 4] # 사진 위치 설정
    ax.imshow(np.transpose(images[idx].numpy(), (1, 2, 0))) # 넘파이 차원 설정
    ax.set_title(class_names[labels[idx]]) # 각 사진에 들어갈 타이틀
    
plt.show()
