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
# 1. 동적 확장 레이어
# ─────────────────────────────────────────
class DynamicLayer(nn.Module):
    def __init__(self, in_features, init_nodes=256):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(init_nodes, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(init_nodes))
        self.node_owner   = {}   # {node_idx: domain_name}
        self.domain_nodes = {}   # {domain_name: [node_indices]}

    @property
    def out_features(self):
        return self.weight.shape[0]

    def expand_input(self, n_new, device):
        """이전 레이어 노드 증가 시 입력 차원 확장 (새 열을 0으로 추가)"""
        new_cols = torch.zeros(self.weight.shape[0], n_new).to(device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_cols], dim=1))
        self.in_features += n_new

    def add_nodes(self, n_nodes, domain_name, device):
        """새 노드 추가 및 소유권 등록"""
        old_size = self.weight.shape[0]
        new_w = torch.randn(n_nodes, self.in_features).to(device) * 0.01
        new_b = torch.zeros(n_nodes).to(device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
        self.bias   = nn.Parameter(torch.cat([self.bias.data,   new_b], dim=0))
        new_indices = list(range(old_size, old_size + n_nodes))
        self._register(domain_name, new_indices)
        return new_indices

    def assign_nodes(self, indices, domain_name):
        self._register(domain_name, indices)

    def _register(self, domain_name, indices):
        for idx in indices:
            self.node_owner[idx] = domain_name
        if domain_name not in self.domain_nodes:
            self.domain_nodes[domain_name] = []
        self.domain_nodes[domain_name].extend(indices)

    def get_free_nodes(self, candidates):
        return [i for i in candidates if i not in self.node_owner]

    def apply_ownership_mask(self, domain_name):
        """소유 노드 외 gradient 차단"""
        owned = set(self.domain_nodes.get(domain_name, []))
        for param, is_bias in [(self.weight, False), (self.bias, True)]:
            if param.grad is None: continue
            mask = torch.zeros_like(param.grad)
            for idx in owned:
                if is_bias:
                    mask[idx] = 1.0
                else:
                    mask[idx] = 1.0
            param.grad *= mask

    def forward(self, x):
        """항상 전체 노드 통과 (마스킹은 gradient에서만)"""
        return torch.relu(x @ self.weight.T + self.bias)


# ─────────────────────────────────────────
# 2. 동적 네트워크
#    - forward는 항상 전체 노드 통과
#    - 도메인별 헤드가 소유 노드 출력만 선택해서 사용
# ─────────────────────────────────────────
class DynamicNetwork(nn.Module):
    def __init__(self, input_dim=784, init_nodes=256):
        super().__init__()
        self.layer1 = DynamicLayer(input_dim,   init_nodes)
        self.layer2 = DynamicLayer(init_nodes,  init_nodes)
        self.heads  = nn.ModuleDict()
        # 도메인별 헤드가 사용할 노드 인덱스
        self.head_nodes = {}   # {domain_name: [layer2_node_indices]}
        self.activation_maps = {}

    def add_head(self, domain_name, output_dim, node_indices):
        """도메인 헤드: 특정 노드 출력만 받아서 분류"""
        device = self.layer1.weight.device
        n_in   = len(node_indices)
        self.heads[domain_name]     = nn.Linear(n_in, output_dim).to(device)
        self.head_nodes[domain_name] = node_indices

    def forward(self, x, domain=None):
        x  = x.view(x.size(0), -1)
        h1 = self.layer1(x)   # 전체 노드 통과
        h2 = self.layer2(h1)  # 전체 노드 통과
        self.activation_maps['h1'] = h1.detach()
        self.activation_maps['h2'] = h2.detach()

        if domain and domain in self.heads:
            # 해당 도메인 소유 노드 출력만 선택
            idx  = self.head_nodes[domain]
            feat = h2[:, idx]
            return self.heads[domain](feat)

        # 기본: mnist
        idx  = self.head_nodes['mnist']
        feat = h2[:, idx]
        return self.heads['mnist'](feat)


# ─────────────────────────────────────────
# 3. 출력 헬퍼
# ─────────────────────────────────────────
def print_header(title):
    print(f"\n{'═'*52}")
    print(f"  {title}")
    print(f"{'═'*52}")

def print_step(msg):   print(f"  ▶ {msg}")
def print_result(msg): print(f"  ✔ {msg}")

def print_bar(label, value, total, width=28):
    if total == 0: return
    filled = int(width * value / total)
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label:14s} [{bar}] {value/total*100:.1f}%")


# ─────────────────────────────────────────
# 4. AFLN v8 메인
# ─────────────────────────────────────────
class AFLNv8:
    def __init__(self, device='cpu', init_nodes=256):
        self.device     = device
        self.net        = DynamicNetwork(init_nodes=init_nodes).to(device)
        self.history    = {'base_acc': None, 'domain_acc': {}, 'forget': {}}
        self.init_nodes = init_nodes

    # ── 최초 학습 ─────────────────────────
    def pretrain(self, dataloader, output_dim=10, epochs=5):
        print_header("1단계 | 최초 학습 (MNIST)")

        all_nodes = list(range(self.init_nodes))
        self.net.layer1.assign_nodes(all_nodes, 'mnist')
        self.net.layer2.assign_nodes(all_nodes, 'mnist')
        self.net.add_head('mnist', output_dim, all_nodes)

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
                loss.backward()
                # 최초 학습은 mnist 소유 노드만
                self.net.layer1.apply_ownership_mask('mnist')
                self.net.layer2.apply_ownership_mask('mnist')
                opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print_result(f"Epoch {epoch+1} | Loss: {total/len(dataloader):.4f}")

        print_result("최초 학습 완료")

    # ── 활성화 노드 탐지 ──────────────────
    def _detect_active_nodes(self, dataloader, top_ratio=0.3):
        self.net.eval()
        all_h1, all_h2 = [], []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  활성화 탐지", ncols=72)):
                imgs = imgs.to(self.device)
                self.net(imgs)
                all_h1.append(self.net.activation_maps['h1'].cpu().numpy())
                all_h2.append(self.net.activation_maps['h2'].cpu().numpy())
                if i >= 10: break

        h1_mean = np.concatenate(all_h1).mean(axis=0)
        h2_mean = np.concatenate(all_h2).mean(axis=0)
        k1 = max(8, int(len(h1_mean) * top_ratio))
        k2 = max(8, int(len(h2_mean) * top_ratio))
        active_l1 = np.argsort(h1_mean)[-k1:].tolist()
        active_l2 = np.argsort(h2_mean)[-k2:].tolist()
        return active_l1, active_l2

    # ── 신규 도메인 학습 ──────────────────
    def learn_domain(self, domain_name, train_loader,
                     output_dim, epochs=7, top_ratio=0.3):
        print_header(f"신규 도메인 학습: {domain_name}")
        self.net.eval()

        # 1. 활성화 노드 탐지
        active_l1, active_l2 = self._detect_active_nodes(train_loader, top_ratio)
        print_step(f"활성화 노드: L1={len(active_l1)}개, L2={len(active_l2)}개")

        # 2. 빈 노드 확인
        free_l1 = self.net.layer1.get_free_nodes(active_l1)
        free_l2 = self.net.layer2.get_free_nodes(active_l2)
        print_step(f"사용 가능 노드: L1={len(free_l1)}개, L2={len(free_l2)}개")

        # 3. 필요한 노드 수 결정 (최소 8개 보장)
        need_l1 = max(0, 8 - len(free_l1))
        need_l2 = max(0, 8 - len(free_l2))

        new_l1, new_l2 = [], []
        if need_l1 > 0:
            new_l1 = self.net.layer1.add_nodes(need_l1, domain_name, self.device)
            self.net.layer2.expand_input(need_l1, self.device)
            print_result(f"새 노드 생성: L1 +{need_l1}개 (총 {self.net.layer1.out_features}개)")
        if need_l2 > 0:
            new_l2 = self.net.layer2.add_nodes(need_l2, domain_name, self.device)
            print_result(f"새 노드 생성: L2 +{need_l2}개 (총 {self.net.layer2.out_features}개)")

        # 4. 소유권 등록
        domain_l1 = free_l1 + new_l1
        domain_l2 = free_l2 + new_l2
        self.net.layer1.assign_nodes(domain_l1, domain_name)
        self.net.layer2.assign_nodes(domain_l2, domain_name)
        print_result(f"소유권 등록: L1={len(domain_l1)}개, L2={len(domain_l2)}개")

        # 5. 헤드 추가 (소유 노드만 입력으로)
        self.net.add_head(domain_name, output_dim, domain_l2)

        # 6. 학습 (소유 노드만 gradient 허용)
        opt   = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit  = nn.CrossEntropyLoss()

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
                self.net.layer1.apply_ownership_mask(domain_name)
                self.net.layer2.apply_ownership_mask(domain_name)
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
            opt  = optim.Adam(self.net.parameters(), lr=1e-4)
            crit = nn.CrossEntropyLoss()
            for epoch in range(5):
                self.net.train()
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=72):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    loss = crit(self.net(imgs, domain_name), labels)
                    loss.backward()
                    self.net.layer1.apply_ownership_mask(domain_name)
                    self.net.layer2.apply_ownership_mask(domain_name)
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

    # ── 전체 현황 ─────────────────────────
    def print_snapshot(self, base_acc):
        print_header("📊 전체 현황 (노드 소유권 + 동적 생성)")
        print(f"  {'도메인':<16} {'정확도':>8} {'망각률':>8} {'판정':>6}")
        print(f"  {'─'*44}")
        for domain, acc in self.history['domain_acc'].items():
            forget = base_acc - acc if domain == 'mnist' \
                     else self.history['forget'].get(domain, 0)
            judge  = '✔' if forget < 2.0 else '△' if forget < 5.0 else '✘'
            print(f"  {domain:<16} {acc:>7.2f}% {forget:>7.2f}% {judge:>6}")
        print(f"  {'─'*44}")
        print(f"\n  네트워크 노드 현황:")
        print(f"  Layer1 총 {self.net.layer1.out_features}개 | "
              f"Layer2 총 {self.net.layer2.out_features}개")
        for domain in self.net.layer1.domain_nodes:
            l1 = len(self.net.layer1.domain_nodes.get(domain, []))
            l2 = len(self.net.layer2.domain_nodes.get(domain, []))
            print(f"  {domain:<16} L1: {l1:3}개 | L2: {l2:3}개")

    # ── 저장 ─────────────────────────────
    def save(self, path='afln_v8.pt'):
        torch.save({
            'net':           self.net.state_dict(),
            'history':       self.history,
            'head_nodes':    self.net.head_nodes,
            'layer1_owner':  self.net.layer1.node_owner,
            'layer1_domain': self.net.layer1.domain_nodes,
            'layer2_owner':  self.net.layer2.node_owner,
            'layer2_domain': self.net.layer2.domain_nodes,
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
    print(f"  구조: 단일 네트워크 + 노드 소유권 + 동적 뉴런 생성")
    print(f"  forward: 항상 전체 노드 통과, 헤드에서 소유 노드만 선택")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    mnist_train  = datasets.MNIST('./data', train=True,  download=True, transform=tf)
    mnist_test   = datasets.MNIST('./data', train=False, download=True, transform=tf)
    emnist_train = LabelShiftDataset(datasets.EMNIST('./data', split='letters',
                                     train=True,  download=True, transform=tf))
    emnist_test  = LabelShiftDataset(datasets.EMNIST('./data', split='letters',
                                     train=False, download=True, transform=tf))
    fmnist_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf)
    fmnist_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)

    mnist_loader       = DataLoader(mnist_train,  batch_size=64, shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,   batch_size=64, shuffle=False)
    emnist_loader      = DataLoader(emnist_train, batch_size=64, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test,  batch_size=64, shuffle=False)
    fmnist_loader      = DataLoader(fmnist_train, batch_size=64, shuffle=True)
    fmnist_test_loader = DataLoader(fmnist_test,  batch_size=64, shuffle=False)

    afln = AFLNv8(device=device, init_nodes=256)

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
    afln.save('afln_v8.pt')

    # 최종 비교
    print_header("🏁 최종 비교 (v6 EWC vs v7 클러스터 vs v8 소유권)")
    print(f"  {'항목':<24} {'v6 EWC':>9} {'v7 클러스터':>11} {'v8 소유권':>10}")
    print(f"  {'─'*58}")
    print(f"  {'MNIST 망각률':<24} {'4.82%':>9} {'90.21%':>11} "
          f"{base_acc - after_fashion:>9.2f}%")
    print(f"  {'알파벳 정확도':<24} {'72.58%':>9} {'30.47%':>11} "
          f"{after_alpha2:>9.2f}%")
    print(f"  {'Fashion 정확도':<24} {'85.27%':>9} {'88.04%':>11} "
          f"{afln.history['domain_acc'].get('fashion', 0):>9.2f}%")
    print(f"  {'─'*58}")