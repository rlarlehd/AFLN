import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# 1. 기본 신경망
# ─────────────────────────────────────────
class BaseNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.activation_maps = {}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        self.activation_maps['h1'] = h1.detach()
        self.activation_maps['h2'] = h2.detach()
        return out


# ─────────────────────────────────────────
# 2. 미니망
# ─────────────────────────────────────────
class MiniNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 3. 진행률 출력 헬퍼
# ─────────────────────────────────────────
def print_header(title):
    print(f"\n{'═'*50}")
    print(f"  {title}")
    print(f"{'═'*50}")

def print_step(msg):
    print(f"  ▶ {msg}")

def print_result(msg):
    print(f"  ✔ {msg}")

def print_bar(label, value, total, width=30):
    filled = int(width * value / total)
    bar = '█' * filled + '░' * (width - filled)
    pct = value / total * 100
    print(f"  {label:10s} [{bar}] {pct:.1f}%")


# ─────────────────────────────────────────
# 4. AFLN 메인 클래스
# ─────────────────────────────────────────
class AFLN:
    def __init__(self, device='cpu'):
        self.device = device
        self.base_net = BaseNetwork().to(device)
        self.clusters = {}
        self.mini_nets = {}
        self.frozen = False
        self.history = {
            'pretrain_loss': [],
            'mini_loss': {},
            'exam_acc': {},
            'mnist_acc_before': None,
            'mnist_acc_after': None,
        }

    # ── 1단계: 최초 학습 ──────────────────
    def pretrain(self, dataloader, epochs=5):
        print_header("1단계 | 최초 학습 (MNIST)")
        optimizer = optim.Adam(self.base_net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.base_net.train()

        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(
                dataloader,
                desc=f"  Epoch {epoch+1}/{epochs}",
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out = self.base_net(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            self.history['pretrain_loss'].append(avg_loss)
            print_result(f"Epoch {epoch+1} 완료 | 평균 Loss: {avg_loss:.4f}")

        print_result("최초 학습 완료 → 클러스터링 시작")
        self._generate_clusters(dataloader)

    # ── DBSCAN 대분류 자율 생성 ────────────
    def _generate_clusters(self, dataloader):
        print_header("클러스터링 | DBSCAN 대분류 자율 생성")
        self.base_net.eval()
        all_activations = []

        print_step("활성화 패턴 수집 중...")
        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  패턴 수집", ncols=70)):
                imgs = imgs.to(self.device)
                self.base_net(imgs)
                acts = self.base_net.activation_maps['h2'].cpu().numpy()
                all_activations.append(acts)
                if i >= 20:
                    break

        activations = np.concatenate(all_activations, axis=0)
        print_result(f"수집된 샘플 수: {len(activations)}")

        print_step("PCA 차원 축소 중...")
        scaler = StandardScaler()
        activations_scaled = scaler.fit_transform(activations)
        pca = PCA(n_components=10)
        activations_reduced = pca.fit_transform(activations_scaled)
        print_result(f"분산 설명률: {pca.explained_variance_ratio_.sum()*100:.1f}%")

        print_step("DBSCAN 클러스터링 중...")
        db = DBSCAN(eps=3.0, min_samples=5)
        labels = db.fit_predict(activations_reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).sum()

        print_result(f"대분류 {n_clusters}개 자율 생성 완료")
        print_result(f"노이즈 포인트: {noise}개 (미분류)")

        # 클러스터 분포 시각화
        print("\n  클러스터 분포:")
        for c in sorted(set(labels)):
            cnt = (labels == c).sum()
            label = "노이즈" if c == -1 else f"클러스터 {c}"
            print_bar(label, cnt, len(labels))

        self.clusters = {
            'labels': labels, 'n_clusters': n_clusters,
            'scaler': scaler, 'pca': pca, 'db': db
        }

        # 기본망 동결
        self.frozen = True
        for param in self.base_net.parameters():
            param.requires_grad = False
        print_result("기존 가중치 완전 동결 ✓")

    # ── 2단계: 신규 학습 ──────────────────
    def adaptive_learn(self, dataloader, output_dim, domain_name):
        print_header(f"2단계 | 신규 학습: {domain_name}")
        self.base_net.eval()
        criterion = nn.CrossEntropyLoss()

        # 복잡도 측정
        print_step("기존 네트워크로 복잡도 측정 중...")
        total_entropy, count = 0, 0
        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc="  복잡도 측정", ncols=70):
                imgs = imgs.to(self.device)
                out = self.base_net(imgs)
                entropy = -torch.sum(
                    torch.softmax(out, dim=1) * torch.log_softmax(out, dim=1), dim=1
                ).mean()
                total_entropy += entropy.item()
                count += 1
                if count > 10:
                    break

        avg_complexity = total_entropy / count

        # 엔트로피 정규화 (0~log(10) 범위 → 0~1)
        max_entropy = np.log(10)  # base_net output_dim=10 기준
        norm = min(avg_complexity / max_entropy, 1.0)

        # 32~256 선형 매핑 + 출력 클래스 수 반영
        mini_hidden = int(32 + norm * (256 - 32))
        mini_hidden = max(mini_hidden, output_dim * 4)  # 클래스 수 기반 하한
        mini_hidden = min(mini_hidden, 512)
        print_result(f"복잡도: {avg_complexity:.4f} (정규화: {norm:.2f}) → 미니망 크기 자동 결정: hidden={mini_hidden}")

        # 미니망 생성
        mini_net = MiniNetwork(256, mini_hidden, output_dim).to(self.device)
        optimizer = optim.Adam(mini_net.parameters(), lr=1e-3)
        self.history['mini_loss'][domain_name] = []

        print_step(f"미니망 학습 시작 (hidden={mini_hidden}, output={output_dim})")
        mini_net.train()
        for epoch in range(5):
            total = 0
            pbar = tqdm(
                dataloader,
                desc=f"  Epoch {epoch+1}/5",
                ncols=70,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    self.base_net(imgs)
                    features = self.base_net.activation_maps['h2']
                optimizer.zero_grad()
                out = mini_net(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = total / len(dataloader)
            self.history['mini_loss'][domain_name].append(avg)
            print_result(f"Epoch {epoch+1} | 평균 Loss: {avg:.4f}")

        self.mini_nets[domain_name] = mini_net
        print_result(f"{domain_name} 미니망 생성 완료 ✓")

    # ── 3단계: 시험 ───────────────────────
    def exam(self, dataloader, domain_name):
        print_header(f"3단계 | 시험: {domain_name}")
        mini_net = self.mini_nets[domain_name]
        criterion = nn.CrossEntropyLoss()

        mini_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  시험 중", ncols=70):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.base_net(imgs)
                features = self.base_net.activation_maps['h2']
                out = mini_net(features)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        self.history['exam_acc'][domain_name] = accuracy
        print_bar("정확도", correct, total)
        print_result(f"시험 정확도: {accuracy:.2f}%")

        if accuracy < 80:
            print_step(f"정확도 부족 ({accuracy:.1f}% < 80%) → 해당 미니망만 파인튜닝")
            optimizer = optim.Adam(mini_net.parameters(), lr=1e-4)
            mini_net.train()
            for epoch in range(5):
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=70):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        self.base_net(imgs)
                        features = self.base_net.activation_maps['h2']
                    optimizer.zero_grad()
                    out = mini_net(features)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
            print_result("파인튜닝 완료 ✓")
        else:
            print_result(f"통과 → 연결 강화 ✓")

        return accuracy

    # ── 망각률 검증 ───────────────────────
    def verify_no_forgetting(self, dataloader, label=""):
        print_header(f"검증 | 망각률 확인 {label}")
        self.base_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  검증 중", ncols=70):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.base_net(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        print_bar("MNIST", correct, total)
        print_result(f"MNIST 정확도: {accuracy:.2f}%")
        return accuracy

    # ── 최종 요약 ─────────────────────────
    def summary(self):
        print_header("AFLN 최종 결과 요약")
        before = self.history['mnist_acc_before']
        after  = self.history['mnist_acc_after']
        forget = before - after if before and after else 0

        print(f"  {'항목':<20} {'값':>10}")
        print(f"  {'─'*32}")
        print(f"  {'MNIST 학습 전 정확도':<20} {before:>9.2f}%")
        print(f"  {'MNIST 학습 후 정확도':<20} {after:>9.2f}%")
        print(f"  {'망각률':<20} {forget:>9.2f}%")
        for domain, acc in self.history['exam_acc'].items():
            print(f"  {domain+' 시험 정확도':<20} {acc:>9.2f}%")
        print(f"  {'생성된 대분류':<20} {self.clusters.get('n_clusters', 0):>9}개")
        print(f"  {'생성된 미니망':<20} {len(self.mini_nets):>9}개")
        print(f"\n  {'─'*32}")
        if forget < 2.0:
            print("  ✔ 망각률 2% 미만 → 가능성 검증 성공! 🎉")
        else:
            print(f"  ✘ 망각률 {forget:.1f}% → 구조 개선 필요")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train  = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    mnist_test   = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # EMNIST letters 레이블은 1~26 → 0~25로 변환 필요
    def emnist_transform_label(item):
        img, label = item
        return img, label - 1

    emnist_train_raw = datasets.EMNIST('./data', split='letters', train=True,  download=True, transform=transform)
    emnist_test_raw  = datasets.EMNIST('./data', split='letters', train=False, download=True, transform=transform)

    from torch.utils.data import Dataset
    class LabelShiftDataset(Dataset):
        def __init__(self, dataset, shift=-1):
            self.dataset = dataset
            self.shift = shift
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            return img, label + self.shift

    emnist_train = LabelShiftDataset(emnist_train_raw, shift=-1)
    emnist_test  = LabelShiftDataset(emnist_test_raw,  shift=-1)

    mnist_loader       = DataLoader(mnist_train,  batch_size=64, shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,   batch_size=64, shuffle=False)
    emnist_loader      = DataLoader(emnist_train, batch_size=64, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test,  batch_size=64, shuffle=False)

    afln = AFLN(device=device)

    # 1단계
    afln.pretrain(mnist_loader, epochs=5)
    afln.history['mnist_acc_before'] = afln.verify_no_forgetting(mnist_test_loader, "(알파벳 학습 전)")

    # 2단계
    afln.adaptive_learn(emnist_loader, output_dim=26, domain_name='alphabet')

    # 3단계
    afln.exam(emnist_test_loader, domain_name='alphabet')

    # 망각률 검증
    afln.history['mnist_acc_after'] = afln.verify_no_forgetting(mnist_test_loader, "(알파벳 학습 후)")

    # 최종 요약
    afln.summary()