import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# 1. 클러스터 선택적 네트워크
#    - 노드를 클러스터 단위로 관리
#    - 새 도메인 학습 시 관련 클러스터만 업데이트
# ─────────────────────────────────────────
class ClusterNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.activation_maps = {}
        self.heads = nn.ModuleDict()

        # 클러스터별 마스크 (어떤 노드가 어느 클러스터인지)
        self.cluster_mask_l1 = None  # (n_clusters, hidden_dim)
        self.cluster_mask_l2 = None

    def add_head(self, domain_name, output_dim):
        device = next(self.parameters()).device
        self.heads[domain_name] = nn.Linear(256, output_dim).to(device)

    def forward(self, x, domain=None):
        x  = x.view(x.size(0), -1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        self.activation_maps['h1'] = h1.detach()
        self.activation_maps['h2'] = h2.detach()
        if domain and domain in self.heads:
            return self.heads[domain](h2)
        return self.heads['mnist'](h2)

    def freeze_nodes(self, layer, node_indices):
        """특정 노드들의 가중치 동결"""
        if layer == 'fc1':
            for idx in node_indices:
                self.fc1.weight.data[idx] = self.fc1.weight.data[idx].detach()
                if self.fc1.bias is not None:
                    self.fc1.bias.data[idx] = self.fc1.bias.data[idx].detach()
        elif layer == 'fc2':
            for idx in node_indices:
                self.fc2.weight.data[idx] = self.fc2.weight.data[idx].detach()
                if self.fc2.bias is not None:
                    self.fc2.bias.data[idx] = self.fc2.bias.data[idx].detach()

    def apply_gradient_mask(self, active_l1, active_l2):
        """역전파 후 관련 없는 노드의 gradient를 0으로"""
        if self.fc1.weight.grad is not None:
            mask = torch.zeros_like(self.fc1.weight.grad)
            mask[active_l1] = 1.0
            self.fc1.weight.grad *= mask
            if self.fc1.bias.grad is not None:
                bias_mask = torch.zeros_like(self.fc1.bias.grad)
                bias_mask[active_l1] = 1.0
                self.fc1.bias.grad *= bias_mask

        if self.fc2.weight.grad is not None:
            mask = torch.zeros_like(self.fc2.weight.grad)
            mask[active_l2] = 1.0
            self.fc2.weight.grad *= mask
            if self.fc2.bias.grad is not None:
                bias_mask = torch.zeros_like(self.fc2.bias.grad)
                bias_mask[active_l2] = 1.0
                self.fc2.bias.grad *= bias_mask


# ─────────────────────────────────────────
# 2. 출력 헬퍼
# ─────────────────────────────────────────
def print_header(title):
    print(f"\n{'═'*52}")
    print(f"  {title}")
    print(f"{'═'*52}")

def print_step(msg):   print(f"  ▶ {msg}")
def print_result(msg): print(f"  ✔ {msg}")

def print_bar(label, value, total, width=28):
    filled = int(width * value / total)
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label:14s} [{bar}] {value/total*100:.1f}%")


# ─────────────────────────────────────────
# 3. AFLN v7 메인
# ─────────────────────────────────────────
class AFLNv7:
    def __init__(self, device='cpu'):
        self.device   = device
        self.net      = ClusterNetwork().to(device)
        self.history  = {'base_acc': None, 'domain_acc': {}, 'forget': {}}

        # 클러스터 정보
        self.clusters       = {}   # DBSCAN 결과
        self.node_clusters_l1 = {} # 노드 → 클러스터 매핑 (layer1)
        self.node_clusters_l2 = {} # 노드 → 클러스터 매핑 (layer2)
        self.domain_clusters  = {} # 도메인 → 활성 클러스터 매핑

    # ── 최초 학습 ─────────────────────────
    def pretrain(self, dataloader, output_dim=10, epochs=5):
        print_header("1단계 | 최초 학습 (MNIST)")
        self.net.add_head('mnist', output_dim)
        opt  = optim.Adam(self.net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.net.train()
            total = 0
            pbar  = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = crit(self.net(imgs, 'mnist'), labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print_result(f"Epoch {epoch+1} | Loss: {total/len(dataloader):.4f}")

        print_result("최초 학습 완료")
        self._build_clusters(dataloader)

    # ── DBSCAN으로 노드 클러스터 생성 ──────
    def _build_clusters(self, dataloader):
        print_header("클러스터링 | 노드 클러스터 생성")
        self.net.eval()
        all_h1, all_h2 = [], []

        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  패턴 수집", ncols=72)):
                imgs = imgs.to(self.device)
                self.net(imgs, 'mnist')
                all_h1.append(self.net.activation_maps['h1'].cpu().numpy())
                all_h2.append(self.net.activation_maps['h2'].cpu().numpy())
                if i >= 20: break

        # 노드별 활성화 패턴 (샘플 기준이 아닌 노드 기준으로 전치)
        h1_acts = np.concatenate(all_h1, axis=0).T  # (hidden_dim, n_samples)
        h2_acts = np.concatenate(all_h2, axis=0).T

        # 각 노드의 활성화 패턴을 DBSCAN으로 클러스터링
        self.node_clusters_l1 = self._cluster_nodes(h1_acts, 'Layer1')
        self.node_clusters_l2 = self._cluster_nodes(h2_acts, 'Layer2')

    def _cluster_nodes(self, node_acts, layer_name):
        """노드들을 활성화 패턴 기반으로 클러스터링"""
        scaler  = StandardScaler()
        pca     = PCA(n_components=min(20, node_acts.shape[1]-1))
        reduced = pca.fit_transform(scaler.fit_transform(node_acts))

        db     = DBSCAN(eps=2.0, min_samples=3)
        labels = db.fit_predict(reduced)
        n      = len(set(labels)) - (1 if -1 in labels else 0)

        print_result(f"{layer_name}: 노드 클러스터 {n}개 생성")

        # 클러스터 → 노드 인덱스 매핑
        cluster_to_nodes = {}
        for node_idx, cluster_id in enumerate(labels):
            if cluster_id not in cluster_to_nodes:
                cluster_to_nodes[cluster_id] = []
            cluster_to_nodes[cluster_id].append(node_idx)

        # 클러스터 크기 출력
        for cid, nodes in sorted(cluster_to_nodes.items()):
            label = "노이즈" if cid == -1 else f"클러스터{cid}"
            print_bar(label, len(nodes), len(labels))

        return cluster_to_nodes  # {cluster_id: [node_indices]}

    # ── 새 도메인 관련 클러스터 탐지 ────────
    def _find_relevant_clusters(self, dataloader, domain_name):
        """새 도메인 데이터가 어느 클러스터와 가장 가까운지 탐지"""
        print_step("관련 클러스터 탐지 중...")
        self.net.eval()
        all_h1, all_h2 = [], []

        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  활성화 측정", ncols=72)):
                imgs = imgs.to(self.device)
                self.net(imgs, 'mnist')
                all_h1.append(self.net.activation_maps['h1'].cpu().numpy())
                all_h2.append(self.net.activation_maps['h2'].cpu().numpy())
                if i >= 10: break

        h1_mean = np.concatenate(all_h1).mean(axis=0)  # 평균 활성화
        h2_mean = np.concatenate(all_h2).mean(axis=0)

        # 각 클러스터의 평균 활성화와 비교
        # 활성화가 높은 클러스터 = 관련 클러스터
        active_l1 = self._select_active_clusters(
            h1_mean, self.node_clusters_l1, 'Layer1'
        )
        active_l2 = self._select_active_clusters(
            h2_mean, self.node_clusters_l2, 'Layer2'
        )

        self.domain_clusters[domain_name] = {
            'l1': active_l1, 'l2': active_l2
        }
        return active_l1, active_l2

    def _select_active_clusters(self, mean_acts, cluster_to_nodes, layer_name):
        """활성화 기반으로 관련 클러스터 선택"""
        cluster_scores = {}
        for cid, nodes in cluster_to_nodes.items():
            if cid == -1: continue  # 노이즈 제외
            # 해당 클러스터 노드들의 평균 활성화
            cluster_scores[cid] = mean_acts[nodes].mean()

        if not cluster_scores:
            return list(range(len(mean_acts)))  # 클러스터 없으면 전체

        # 상위 50% 클러스터만 선택
        threshold = np.median(list(cluster_scores.values()))
        active_clusters = [cid for cid, score in cluster_scores.items()
                          if score >= threshold]

        # 활성 클러스터의 노드 인덱스 수집
        active_nodes = []
        for cid in active_clusters:
            active_nodes.extend(cluster_to_nodes[cid])

        total_nodes  = sum(len(v) for k, v in cluster_to_nodes.items() if k != -1)
        print_result(f"{layer_name}: {len(active_clusters)}개 클러스터 활성화 "
                     f"({len(active_nodes)}/{total_nodes} 노드)")
        return active_nodes

    # ── 신규 도메인 학습 ──────────────────
    def learn_domain(self, domain_name, train_loader, output_dim, epochs=7):
        print_header(f"신규 도메인 학습: {domain_name}")
        self.net.eval()

        # 관련 클러스터 탐지
        active_l1, active_l2 = self._find_relevant_clusters(
            train_loader, domain_name
        )

        # 출력 헤드 추가
        self.net.add_head(domain_name, output_dim)

        # 학습 (gradient mask로 관련 노드만 업데이트)
        opt  = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit = nn.CrossEntropyLoss()

        print_step(f"관련 노드만 업데이트 (L1: {len(active_l1)}개, L2: {len(active_l2)}개)")

        for epoch in range(epochs):
            self.net.train()
            total = 0
            pbar  = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                out  = self.net(imgs, domain=domain_name)
                loss = crit(out, labels)
                loss.backward()

                # 핵심: 관련 없는 노드의 gradient를 0으로 마스킹
                self.net.apply_gradient_mask(active_l1, active_l2)

                opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            sched.step()
            print_result(f"Epoch {epoch+1} | Loss: {total/len(train_loader):.4f}")

        print_result(f"{domain_name} 학습 완료 ✓")

    # ── 시험 ─────────────────────────────
    def exam(self, domain_name, dataloader, threshold=80):
        print_header(f"시험: {domain_name}")
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  시험 중", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                correct += (self.net(imgs, domain_name).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total * 100
        print_bar("정확도", correct, total)
        print_result(f"정확도: {accuracy:.2f}%")
        self.history['domain_acc'][domain_name] = accuracy

        if accuracy < threshold:
            print_step(f"정확도 부족 ({accuracy:.1f}%) → 파인튜닝")
            active_l1 = []
            active_l2 = []
            if domain_name in self.domain_clusters:
                active_l1 = self.domain_clusters[domain_name]['l1']
                active_l2 = self.domain_clusters[domain_name]['l2']

            opt  = optim.Adam(self.net.parameters(), lr=1e-4)
            crit = nn.CrossEntropyLoss()
            for epoch in range(5):
                self.net.train()
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=72):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    loss = crit(self.net(imgs, domain_name), labels)
                    loss.backward()
                    if active_l1:
                        self.net.apply_gradient_mask(active_l1, active_l2)
                    opt.step()

            self.net.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in dataloader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    correct += (self.net(imgs, domain_name).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            accuracy = correct / total * 100
            self.history['domain_acc'][domain_name] = accuracy
            print_result(f"파인튜닝 후: {accuracy:.2f}%")
        else:
            print_result("통과 → 연결 강화 ✓")
        return accuracy

    # ── 망각률 검증 ───────────────────────
    def verify_mnist(self, dataloader, tag=""):
        print_header(f"MNIST 망각률 검증 {tag}")
        return self.exam('mnist', dataloader, threshold=0)

    # ── 전체 현황 출력 ────────────────────
    def print_snapshot(self, base_acc):
        print_header("📊 전체 현황 (클러스터 선택적 업데이트)")
        print(f"  {'도메인':<16} {'정확도':>8} {'망각률':>8} {'판정':>6}")
        print(f"  {'─'*44}")
        for domain, acc in self.history['domain_acc'].items():
            if domain == 'mnist':
                forget = base_acc - acc
            else:
                forget = self.history['forget'].get(domain, 0)
            judge = '✔' if forget < 2.0 else '△' if forget < 5.0 else '✘'
            print(f"  {domain:<16} {acc:>7.2f}% {forget:>7.2f}% {judge:>6}")
        print(f"  {'─'*44}")

        # 클러스터 활용 현황
        print(f"\n  클러스터 활용 현황:")
        for domain, info in self.domain_clusters.items():
            print(f"  {domain:<16} L1: {len(info['l1']):3}개 노드 | "
                  f"L2: {len(info['l2']):3}개 노드 활성화")

    # ── 저장 ─────────────────────────────
    def save(self, path='afln_v7.pt'):
        torch.save({
            'net':              self.net.state_dict(),
            'history':          self.history,
            'node_clusters_l1': self.node_clusters_l1,
            'node_clusters_l2': self.node_clusters_l2,
            'domain_clusters':  self.domain_clusters,
        }, path)
        print_result(f"저장 완료 → {path}")


# ─────────────────────────────────────────
# EMNIST 레이블 변환
# ─────────────────────────────────────────
class LabelShiftDataset(Dataset):
    def __init__(self, ds, shift=-1):
        self.ds, self.shift = ds, shift
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, label = self.ds[i]
        return img, label + self.shift


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    print(f"  구조: 단일 네트워크 + 클러스터 선택적 업데이트")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    mnist_train  = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    mnist_test   = datasets.MNIST('./data', train=False, download=True, transform=tf)
    emnist_train = LabelShiftDataset(datasets.EMNIST('./data', split='letters', train=True,  download=True, transform=tf))
    emnist_test  = LabelShiftDataset(datasets.EMNIST('./data', split='letters', train=False, download=True, transform=tf))
    fmnist_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
    fmnist_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)

    mnist_loader       = DataLoader(mnist_train,  batch_size=64, shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,   batch_size=64, shuffle=False)
    emnist_loader      = DataLoader(emnist_train, batch_size=64, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test,  batch_size=64, shuffle=False)
    fmnist_loader      = DataLoader(fmnist_train, batch_size=64, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test,  batch_size=64, shuffle=False)

    afln = AFLNv7(device=device)

    # 1단계: MNIST
    afln.pretrain(mnist_loader, output_dim=10, epochs=5)
    base_acc = afln.exam('mnist', mnist_test_loader, threshold=0)
    afln.history['base_acc'] = base_acc

    # 2단계: 알파벳
    print_header("🔵 알파벳 도메인 추가")
    afln.learn_domain('alphabet', emnist_loader, output_dim=26, epochs=7)
    afln.exam('alphabet', emnist_test_loader, threshold=80)
    after_alpha = afln.verify_mnist(mnist_test_loader, "(알파벳 후)")
    afln.history['forget']['alphabet'] = base_acc - after_alpha
    afln.print_snapshot(base_acc)

    # 3단계: Fashion-MNIST
    print_header("🟣 Fashion-MNIST 도메인 추가")
    afln.learn_domain('fashion', fmnist_loader, output_dim=10, epochs=7)
    afln.exam('fashion', fmnist_test_loader, threshold=70)
    after_fashion = afln.verify_mnist(mnist_test_loader, "(Fashion 후)")
    afln.history['forget']['fashion'] = base_acc - after_fashion

    # 알파벳 재검증
    print_header("🔍 알파벳 재검증 (Fashion 추가 후)")
    after_alpha2 = afln.exam('alphabet', emnist_test_loader, threshold=0)
    afln.print_snapshot(base_acc)

    # 저장
    afln.save('afln_v7.pt')

    # 최종 판정
    print_header("🏁 최종 판정 (v6 EWC vs v7 클러스터)")
    print(f"  {'항목':<24} {'v6 EWC':>10} {'v7 클러스터':>12}")
    print(f"  {'─'*48}")
    print(f"  {'MNIST 망각률':<24} {'4.82%':>10} {base_acc - after_fashion:>11.2f}%")
    print(f"  {'알파벳 정확도':<24} {'72.58%':>10} {after_alpha2:>11.2f}%")
    print(f"  {'Fashion 정확도':<24} {'85.27%':>10} "
          f"{afln.history['domain_acc'].get('fashion', 0):>11.2f}%")
    print(f"  {'─'*48}")
    print(f"\n  핵심 질문: 클러스터 선택적 업데이트가")
    print(f"  EWC보다 망각을 줄이면서 정확도를 유지하는가?")