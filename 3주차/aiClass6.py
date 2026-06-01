import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# 1. 계산 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# [요청 1] 원-핫 인코딩을 적용한 맞춤형 데이터셋 클래스 정의
# ---------------------------------------------------------
class OneHotMNIST(Dataset):
    # 클래스를 처음 만들 때 실행되는 설정 단계입니다.
    def __init__(self, root, train=True, transform=None, download=False):
        # 파이토치에서 제공하는 기본 MNIST 데이터셋을 불러와 내부에 저장합니다.
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=transform)
        self.num_classes = 10           # 분류할 숫자의 개수 (0~9까지 총 10개)

    # 데이터셋에서 "n번째 데이터를 주세요"라고 할 때 호출되는 함수입니다.
    def __getitem__(self, index):
        image, target = self.mnist[index] # 기본 데이터셋에서 이미지와 정수 정답(예: 3)을 꺼냅니다.
        
        # [원-핫 인코딩 과정]
        # 1. 먼저 모든 값이 0인 길이가 10인 리스트(텐서)를 만듭니다. [0,0,0,0,0,0,0,0,0,0]
        one_hot_target = torch.zeros(self.num_classes)
        
        # 2. 정답에 해당하는 위치만 1.0으로 바꿉니다. (예: target이 3이면 3번 인덱스를 1로)
        # 결과 예: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        one_hot_target[target] = 1.0
        
        return image, one_hot_target     # 이미지와 함께 변환된 원-핫 정답을 반환합니다.

    # 데이터셋의 전체 개수가 몇 개인지 알려주는 함수입니다 (MNIST는 보통 6만 개).
    def __len__(self):
        return len(self.mnist)

# 2. 이미지 전처리(요리 준비) 및 데이터 로더(배급망) 설정

# 여러 가지 전처리 단계를 하나로 묶어주는 꾸러미입니다.
transform = transforms.Compose([
    transforms.ToTensor(),                # 1. 일반 이미지를 모델이 계산할 수 있는 숫자 행렬(Tensor)로 바꿉니다.
    transforms.Normalize((0.1307,), (0.3081,)) # 2. 데이터를 일정한 범위로 맞춥니다 (학습 속도와 성능이 좋아집니다).
])

# [요청하신 원-핫 인코딩이 적용된] 데이터셋 인스턴스를 생성합니다.
# root: 데이터 저장 위치, train: 학습용인지 여부, download: 없으면 인터넷에서 받을지, transform: 위에서 정한 전처리 적용
train_dataset = OneHotMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = OneHotMNIST(root='./data', train=False, transform=transform)

# DataLoader는 모델이 학습할 때 데이터를 효율적으로 꺼내갈 수 있게 돕는 도구입니다.
# batch_size: 한 번에 몇 개의 이미지를 묶어서 보여줄지 결정 (여기서는 64개씩)
# shuffle: 데이터를 무작위로 섞을지 결정 (학습할 때는 순서를 외우지 못하게 섞는 게 좋습니다)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 3. CNN 모델 설계도 (이전과 동일)
class ConvNet(nn.Module):
    def __init__(self):
            super(ConvNet, self).__init__()
            # 1. 첫 번째 컨볼루션 층: 1개의 채널(흑백)에서 32개의 특징 지도를 만듭니다.
            # kernel_size=3은 3x3 크기의 돋보기로 이미지를 훑겠다는 뜻입니다.
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1) #(앞의 1은 입력받는 이미지가 흑백이기 때문에, 1을 입력. color 이미지의 경우 RGB를 쓰기때문에 3을 입력. convolution filter를 32종류를 만들고, 이를 적용하여 32개의 특징이 추출된 이미지를 만들겠다는 뜻)
            
            # 2. 두 번째 컨볼루션 층: 32개의 특징을 조합해 64개의 더 고차원적인 특징을 찾습니다.
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
            
            # 3. 드롭아웃: 학습 시 뉴런을 일부 꺼서 모델이 특정 데이터에만 집착하는 것을 막습니다.
            self.dropout1 = nn.Dropout(0.25) # 25%의 노드를 무작위로 쉽니다.
            self.dropout2 = nn.Dropout(0.5)  # 50%의 노드를 무작위로 쉽니다.
            
            # 4. 첫 번째 전결합 층(Fully Connected): 이미지를 한 줄로 펴서(Flatten) 연산합니다.
            # 9216이 나오는 이유: 28x28 이미지가 conv층과 pooling을 거치며 크기가 줄어든 결과물입니다.
            self.fc1 = nn.Linear(9216, 128) 
            
            # 5. 마지막 전결합 층: 128개의 정보를 10개의 숫자(0~9)로 최종 압축합니다.
            # 이 결과값이 원-핫 인코딩된 정답지와 비교될 대상입니다.
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #컨볼루션 층을 쌓고
        x = F.relu(self.conv2(x)) #다시 컨볼루션 층을 쌓고
        x = F.max_pool2d(x, 2) #그 뒤에 풀링층을 쌓고
        x = self.dropout1(x) #그 뒤에 dropout층을 적용하고
        x = torch.flatten(x, 1) #그 뒤에 평탄화 한 뒤에, 
        x = F.relu(self.fc1(x)) #입력층 + Relu 활성함수 적용하고, 
        x = self.dropout2(x) #그 뒤에 dropout층을 적용하고
        x = self.fc2(x) #입력층 + 리니어 활성함수 적용.
        # 중요: 원-핫 인코딩 타겟과 BCEWithLogitsLoss를 사용할 때는
        # 마지막 레이어에 Softmax를 적용하지 않고 '로짓(logits)' 그대로 반환합니다.
        return x

model = ConvNet().to(device) #앞서 정의한 class로 모델 사용


# 4. 최적화 도구 및 손실 함수 설정
# 원-핫 인코딩된 멀티 클래스 분류에는 BCEWithLogitsLoss가 적합합니다.
criterion = nn.BCEWithLogitsLoss() #손실함수 정의
optimizer = optim.Adam(model.parameters(), lr=0.001) #가중치 최적화 알고리즘 정의


# 5. 학습 함수 정의
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()                                   # 모델을 '학습 모드'로 설정 (드롭아웃 등이 활성화됨)
    total_loss = 0                                  # 한 에폭 동안 발생하는 총 손실을 저장할 변수
    
    # train_loader에서 배치(64개 묶음) 단위로 이미지(data)와 정답(target)을 꺼내옵니다.
    for batch_idx, (data, target) in enumerate(train_loader):
        # 데이터와 정답을 계산 장치(GPU 또는 CPU)로 보냅니다.
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()                       # 1. 지난번 계산한 기울기(오차 수정값)를 0으로 초기화
        output = model(data)                        # 2. 모델에 이미지를 넣어 예측값(로짓)을 얻음
        
        # [원-핫 타겟 대응] 예측값과 실제 정답(원-핫 벡터) 사이의 오차를 계산합니다.
        loss = criterion(output, target)            # 3. 손실 함수로 모델이 얼마나 틀렸는지 측정
        loss.backward()                             # 4. 역전파: 오차를 줄이기 위해 각 신경망이 고쳐야 할 양 계산
        optimizer.step()                            # 5. 계산된 값을 바탕으로 모델의 가중치(실력)를 실제로 업데이트
        
        total_loss += loss.item()                   # 시각화를 위해 현재 배치의 오차를 합산
        
        # 200번 배치마다 학습 진행 상황(에폭, 진행률, 현재 오차)을 화면에 출력합니다.
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f}')
                  
    return total_loss / len(train_loader)           # 평균 손실값을 반환하여 학습 상태를 추적하게 함

# 6. 실제 학습 실행
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)

print("\nTraining Finished!")


# ---------------------------------------------------------
# [요청 2] 테스트 이미지 10개에 대한 예측 시각화 함수 정의
# ---------------------------------------------------------
def visualize_predictions(model, device, test_dataset, num_images=10):
    model.eval() # 모델을 평가 모드로 전환
    
    # 시각화할 이미지를 무작위로 선택하기 위한 인덱스 생성
    indices = np.random.choice(len(test_dataset), num_images, replace=False)
    
    # 플롯 설정
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle('Model Predictions on Test Images', fontsize=16)
    axes = axes.ravel() # 2x5 배열을 1차원으로 폅니다.

    with torch.no_grad(): # 예측 시에는 기울기 계산 안 함
        for i, idx in enumerate(indices):
            image, one_hot_target = test_dataset[idx]
            original_image = image.squeeze().numpy() # 시각화를 위해 차원 축소 및 numpy 변환
            
            # 모델 입력 준비 (배치 차원 추가 및 장치 이동)
            input_image = image.unsqueeze(0).to(device)
            output = model(input_image)
            
            # 로짓(logits) 결과에 Softmax를 적용하여 확률 값으로 변환
            probabilities = F.softmax(output, dim=1)
            
            # 가장 높은 확률을 가진 인덱스를 예측값으로 선택
            pred_class = probabilities.argmax(dim=1, keepdim=True).item()
            
            # 원-핫 타겟에서 실제 정답(class) 추출
            true_class = one_hot_target.argmax().item()

            # 이미지 출력
            axes[i].imshow(original_image, cmap='gray')
            
            # 예측값과 실제값 비교하여 제목 표시 (맞으면 초록색, 틀리면 빨간색)
            title_color = 'green' if pred_class == true_class else 'red'
            axes[i].set_title(f'Pred: {pred_class} (True: {true_class})', color=title_color)
            axes[i].axis('off') # 축 숨기기

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 제목과 이미지 사이 간격 조정
    plt.show()

# 학습된 모델로 시각화 함수 실행
visualize_predictions(model, device, test_dataset)
