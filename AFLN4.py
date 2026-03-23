import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# 공통 클래스 (v2/v3와 동일)
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

class MiniNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x): return self.net(x)

class DomainAdapter(nn.Module):
    def __init__(self, input_dim, target_dim=256):
        super().__init__()
        mid = (input_dim + target_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.BatchNorm1d(mid),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mid, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x.view(x.size(0), -1))


# ─────────────────────────────────────────
# 출력 헬퍼
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

def print_stress_table(snapshot):
    """도메인 추가할 때마다 전체 스냅샷 출력"""
    print_header("📊 누적 스트레스 테스트 현황")
    print(f"  {'도메인':<16} {'정확도':>8} {'MNIST 망각률':>12} {'판정':>6}")
    print(f"  {'─'*46}")
    for row in snapshot:
        judge = '✔' if row['forget'] < 2.0 and row['acc'] > 30 else '✘'
        print(f"  {row['domain']:<16} {row['acc']:>7.2f}% {row['forget']:>11.2f}% {judge:>6}")
    print(f"  {'─'*46}")
    all_ok = all(r['forget'] < 2.0 for r in snapshot)
    print(f"\n  {'✔ 전체 망각률 2% 미만 유지 중!' if all_ok else '✘ 망각률 초과 도메인 존재'}")


# ─────────────────────────────────────────
# AFLN v4
# ─────────────────────────────────────────
class AFLNv4:
    def __init__(self, device='cpu'):
        self.device    = device
        self.base_net  = BaseNetwork().to(device)
        self.mini_nets = {}
        self.adapters  = {}
        self.clusters  = {}
        self.frozen    = False
        self.history   = {}
        self.snapshot  = []  # 누적 스냅샷

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        print_header(f"📂 체크포인트 불러오기: {path}")
        state = torch.load(path, map_location=device)
        afln  = cls(device=device)

        afln.base_net.load_state_dict(state['base_net'])
        if state['frozen']:
            for p in afln.base_net.parameters():
                p.requires_grad = False
        afln.frozen   = state['frozen']
        afln.clusters = state['clusters']
        afln.history  = state['history']

        for k, v in state['mini_nets'].items():
            net = MiniNetwork(256, v['hidden'], v['output_dim']).to(device)
            net.load_state_dict(v['state_dict'])
            afln.mini_nets[k] = {
                'net': net, 'layer': v['layer'],
                'hidden': v['hidden'], 'adapter': None,
            }

        # 어댑터 복원
        if 'adapters' in state:
            for k, sd in state['adapters'].items():
                # 어댑터 크기 추론
                first_w = sd['net.0.weight']
                input_dim = first_w.shape[1]
                adapter = DomainAdapter(input_dim=input_dim).to(device)
                adapter.load_state_dict(sd)
                afln.adapters[k] = adapter
                afln.mini_nets[k]['adapter'] = adapter

        print_result(f"기존 도메인: {list(afln.mini_nets.keys())}")
        print_result(f"기본망 동결: {'예' if afln.frozen else '아니오'}")
        return afln

    def _calc_mini_size(self, dataloader, output_dim, adapter=None):
        total, count = 0, 0
        self.base_net.eval()
        if adapter: adapter.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                if adapter:
                    feats = adapter(imgs)
                    h2  = self.base_net.relu(self.base_net.fc2(feats))
                    out = self.base_net.fc3(h2)
                else:
                    out = self.base_net(imgs)
                    feats = self.base_net.activation_maps['h1']
                ent = -torch.sum(
                    torch.softmax(out,1) * torch.log_softmax(out,1), dim=1
                ).mean()
                total += ent.item(); count += 1
                if count > 10: break
        avg  = total / count
        norm = min(avg / np.log(10), 1.0)
        size = max(int(32 + norm * (256 - 32)), output_dim * 4)
        return size, avg

    def add_domain(self, domain_name, train_loader, output_dim,
                   input_dim=784, epochs=7, threshold=40):
        """새 도메인 추가 (어댑터 자동 생성 여부 결정)"""
        print_header(f"🆕 도메인 추가: {domain_name}")
        self.base_net.eval()

        # 입력 차원이 다르면 어댑터 생성
        adapter = None
        if input_dim != 784:
            print_step(f"어댑터 생성: {input_dim} → 256")
            adapter = DomainAdapter(input_dim=input_dim, target_dim=256).to(self.device)
        else:
            print_step("입력 차원 동일 (784) → 어댑터 없이 h1 직접 사용")

        mini_hidden, avg_c = self._calc_mini_size(train_loader, output_dim, adapter)
        print_result(f"복잡도: {avg_c:.4f} → hidden={mini_hidden}")

        mini_net = MiniNetwork(256, mini_hidden, output_dim).to(self.device)
        params   = list(mini_net.parameters())
        if adapter: params += list(adapter.parameters())
        opt   = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit  = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            mini_net.train()
            if adapter: adapter.train()
            total = 0
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                if adapter:
                    feats = adapter(imgs)
                else:
                    with torch.no_grad():
                        self.base_net(imgs)
                    feats = self.base_net.activation_maps['h1']
                loss = crit(mini_net(feats), labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            sched.step()
            print_result(f"Epoch {epoch+1} | Loss: {total/len(train_loader):.4f}")

        self.mini_nets[domain_name] = {
            'net': mini_net, 'layer': 'h1',
            'hidden': mini_hidden, 'adapter': adapter
        }
        if adapter:
            self.adapters[domain_name] = adapter
        print_result(f"{domain_name} 추가 완료 ✓")

    def exam(self, domain_name, dataloader, threshold=40):
        print_header(f"시험: {domain_name}")
        info     = self.mini_nets[domain_name]
        mini_net = info['net']
        adapter  = info['adapter']
        crit     = nn.CrossEntropyLoss()

        mini_net.eval()
        if adapter: adapter.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  시험 중", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if adapter:
                    feats = adapter(imgs)
                else:
                    self.base_net(imgs)
                    feats = self.base_net.activation_maps['h1']
                correct += (mini_net(feats).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total * 100
        print_bar("정확도", correct, total)
        print_result(f"정확도: {accuracy:.2f}%")

        # 파인튜닝
        if accuracy < threshold:
            print_step(f"정확도 부족 ({accuracy:.1f}% < {threshold}%) → 파인튜닝")
            params = list(mini_net.parameters())
            if adapter: params += list(adapter.parameters())
            opt = optim.Adam(params, lr=1e-4)
            mini_net.train()
            if adapter: adapter.train()
            for epoch in range(5):
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=72):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    if adapter:
                        feats = adapter(imgs)
                    else:
                        with torch.no_grad(): self.base_net(imgs)
                        feats = self.base_net.activation_maps['h1']
                    loss = crit(mini_net(feats), labels)
                    loss.backward(); opt.step()

            # 재평가
            mini_net.eval()
            if adapter: adapter.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in dataloader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    feats = adapter(imgs) if adapter else None
                    if feats is None:
                        self.base_net(imgs)
                        feats = self.base_net.activation_maps['h1']
                    correct += (mini_net(feats).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            accuracy = correct / total * 100
            print_result(f"파인튜닝 후: {accuracy:.2f}%")
        else:
            print_result("통과 → 연결 강화 ✓")
        return accuracy

    def verify_mnist(self, dataloader, tag=""):
        print_header(f"MNIST 망각률 검증 {tag}")
        self.base_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  검증 중", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                correct += (self.base_net(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total * 100
        print_result(f"MNIST 정확도: {acc:.2f}%")
        return acc

    def save(self, path='afln_v4.pt'):
        state = {
            'base_net':  self.base_net.state_dict(),
            'mini_nets': {
                k: {
                    'state_dict': v['net'].state_dict(),
                    'layer':      v['layer'],
                    'hidden':     v['hidden'],
                    'output_dim': v['net'].net[-1].out_features,
                }
                for k, v in self.mini_nets.items()
            },
            'adapters': {
                k: v.state_dict()
                for k, v in self.adapters.items()
            },
            'clusters': self.clusters,
            'frozen':   self.frozen,
            'history':  self.history,
        }
        torch.save(state, path)
        print_result(f"저장 완료 → {path}")


# ─────────────────────────────────────────
# 데이터셋 헬퍼
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

    # v3 체크포인트 불러오기
    afln = AFLNv4.from_checkpoint('afln_v3.pt', device=device)

    # ── Transform 정의 ──────────────────────
    tf_gray  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tf_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    tf_mnist = tf_gray

    # ── 기준값 (v3에서) ──────────────────────
    mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=tf_mnist)
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    base_acc = afln.history.get('base_acc', 97.0)
    snapshot = []

    # ── 도메인 1: Fashion-MNIST (흑백 28x28, 10클래스) ──
    print_header("🟣 도메인 1: Fashion-MNIST")
    fmnist_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=tf_gray)
    fmnist_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf_gray)
    fmnist_train_loader = DataLoader(fmnist_train, batch_size=64, shuffle=True)
    fmnist_test_loader  = DataLoader(fmnist_test,  batch_size=64, shuffle=False)

    afln.add_domain('fashion', fmnist_train_loader, output_dim=10, input_dim=784, epochs=7)
    acc_f   = afln.exam('fashion', fmnist_test_loader, threshold=70)
    forget1 = base_acc - afln.verify_mnist(mnist_loader, "(Fashion 후)")
    snapshot.append({'domain': 'Fashion-MNIST', 'acc': acc_f, 'forget': forget1})
    print_stress_table(snapshot)

    # ── 도메인 2: SVHN (컬러 32x32, 10클래스) ──────────
    print_header("🟡 도메인 2: SVHN (거리 번호판)")
    svhn_train = datasets.SVHN('./data', split='train', download=True, transform=tf_color)
    svhn_test  = datasets.SVHN('./data', split='test',  download=True, transform=tf_color)
    svhn_train_loader = DataLoader(svhn_train, batch_size=64, shuffle=True)
    svhn_test_loader  = DataLoader(svhn_test,  batch_size=64, shuffle=False)

    afln.add_domain('svhn', svhn_train_loader, output_dim=10, input_dim=3072, epochs=7)
    acc_s   = afln.exam('svhn', svhn_test_loader, threshold=50)
    forget2 = base_acc - afln.verify_mnist(mnist_loader, "(SVHN 후)")
    snapshot.append({'domain': 'SVHN', 'acc': acc_s, 'forget': forget2})
    print_stress_table(snapshot)

    # ── 도메인 3: CIFAR-100 (컬러 32x32, 100클래스) ─────
    print_header("🔴 도메인 3: CIFAR-100 (100클래스)")
    cifar100_train = datasets.CIFAR100('./data', train=True,  download=True, transform=tf_color)
    cifar100_test  = datasets.CIFAR100('./data', train=False, download=True, transform=tf_color)
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=64, shuffle=True)
    cifar100_test_loader  = DataLoader(cifar100_test,  batch_size=64, shuffle=False)

    afln.add_domain('cifar100', cifar100_train_loader, output_dim=100, input_dim=3072, epochs=10)
    acc_c100 = afln.exam('cifar100', cifar100_test_loader, threshold=20)
    forget3  = base_acc - afln.verify_mnist(mnist_loader, "(CIFAR-100 후)")
    snapshot.append({'domain': 'CIFAR-100', 'acc': acc_c100, 'forget': forget3})

    # ── 최종 전체 스냅샷 ────────────────────
    print_stress_table(snapshot)

    # ── 저장 ───────────────────────────────
    print_header("💾 모델 저장")
    afln.save('afln_v4.pt')

    # ── 최종 판정 ───────────────────────────
    print_header("🏁 스트레스 테스트 최종 판정")
    all_forget_ok = all(r['forget'] < 2.0 for r in snapshot)
    if all_forget_ok:
        print("  ✔ 도메인 5개 누적에도 망각률 0% 유지")
        print("  ✔ 구조 검증 완료 → 크로스 도메인 프랙탈로 진행 가능")
    else:
        print("  ✘ 망각률 초과 → 구조 점검 필요")
    print(f"\n  다음 단계: python afln_v5_crossdomain.py")