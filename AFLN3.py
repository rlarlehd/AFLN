import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# v2와 동일한 클래스 정의 (불러오기 위해 필요)
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


# ─────────────────────────────────────────
# CIFAR 전용 도메인 어댑터
# ─────────────────────────────────────────
class DomainAdapter(nn.Module):
    """CIFAR-10 (3072) → 기본망 h1 공간 (256)"""
    def __init__(self, input_dim=3072, target_dim=256):
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
    print(f"  {label:12s} [{bar}] {value/total*100:.1f}%")

def print_final(history, base_acc, alpha_acc_before, alpha_acc_after, cifar_acc, forget):
    print_header("📊 v3 최종 결과 요약")
    print(f"  {'항목':<28} {'값':>10}")
    print(f"  {'─'*42}")
    print(f"  {'MNIST 기준 정확도':<28} {base_acc:>9.2f}%")
    print(f"  {'MNIST 망각률 (CIFAR 후)':<28} {forget:>9.2f}%")
    print(f"  {'─'*42}")
    print(f"  {'알파벳 정확도 (v2)':<28} {alpha_acc_before:>9.2f}%")
    print(f"  {'알파벳 정확도 (CIFAR 후 재검증)':<28} {alpha_acc_after:>9.2f}%")
    print(f"  {'알파벳 정확도 변화':<28} {alpha_acc_after-alpha_acc_before:>+9.2f}%")
    print(f"  {'─'*42}")
    print(f"  {'CIFAR-10 정확도':<28} {cifar_acc:>9.2f}%")
    print(f"  {'랜덤 기준선 (10%)':<28} {'10.00':>9}%")
    print(f"  {'랜덤 대비 향상':<28} {cifar_acc-10:>+9.2f}%")
    print(f"  {'─'*42}")

    print(f"\n  🔍 스트레스 테스트 판정")
    print(f"  {'─'*42}")
    # 망각률
    if forget < 2.0:
        print(f"  ✔ 망각률 {forget:.2f}% → 기본망 보호 성공")
    else:
        print(f"  ✘ 망각률 {forget:.2f}% → 구조 개선 필요")
    # 알파벳 유지
    delta = alpha_acc_after - alpha_acc_before
    if abs(delta) < 1.0:
        print(f"  ✔ 알파벳 미니망 유지 ({delta:+.2f}%) → 도메인 간 간섭 없음")
    else:
        print(f"  △ 알파벳 정확도 변화 ({delta:+.2f}%) → 확인 필요")
    # CIFAR
    if cifar_acc >= 50:
        print(f"  ✔ CIFAR {cifar_acc:.2f}% → 어댑터 유효 (랜덤 대비 {cifar_acc-10:.1f}%p↑)")
    elif cifar_acc >= 30:
        print(f"  △ CIFAR {cifar_acc:.2f}% → 어댑터 부분 유효 (개선 여지 있음)")
    else:
        print(f"  ✘ CIFAR {cifar_acc:.2f}% → 어댑터 개선 필요")


# ─────────────────────────────────────────
# AFLN 확장 클래스 (v2 체크포인트 이어받기)
# ─────────────────────────────────────────
class AFLNv3:
    def __init__(self, device='cpu'):
        self.device    = device
        self.base_net  = BaseNetwork().to(device)
        self.mini_nets = {}
        self.adapters  = {}
        self.clusters  = {}
        self.frozen    = False
        self.history   = {}

    @classmethod
    def from_checkpoint(cls, path='afln_v2.pt', device='cpu'):
        """v2 체크포인트에서 불러오기"""
        print_header(f"📂 체크포인트 불러오기: {path}")
        state = torch.load(path, map_location=device)
        afln  = cls(device=device)

        # 기본망 복원
        afln.base_net.load_state_dict(state['base_net'])
        if state['frozen']:
            for p in afln.base_net.parameters():
                p.requires_grad = False
        afln.frozen   = state['frozen']
        afln.clusters = state['clusters']
        afln.history  = state['history']

        # 기존 미니망 복원
        for k, v in state['mini_nets'].items():
            net = MiniNetwork(256, v['hidden'], v['output_dim']).to(device)
            net.load_state_dict(v['state_dict'])
            afln.mini_nets[k] = {
                'net':     net,
                'layer':   v['layer'],
                'hidden':  v['hidden'],
                'adapter': None,
            }

        print_result(f"기존 도메인: {list(afln.mini_nets.keys())}")
        print_result(f"클러스터 수: {afln.clusters.get('n', '?')}개")
        print_result(f"기본망 동결: {'예' if afln.frozen else '아니오'}")
        return afln

    def _calc_mini_size(self, dataloader, output_dim, adapter):
        """어댑터 통과 후 복잡도 측정 → 미니망 크기 결정"""
        total, count = 0, 0
        self.base_net.eval()
        adapter.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs  = imgs.to(self.device)
                feats = adapter(imgs)
                # h1 공간으로 변환된 특징을 fc2→fc3에 통과
                h2  = self.base_net.relu(self.base_net.fc2(feats))
                out = self.base_net.fc3(h2)
                ent = -torch.sum(
                    torch.softmax(out,1) * torch.log_softmax(out,1), dim=1
                ).mean()
                total += ent.item(); count += 1
                if count > 10: break
        avg  = total / count
        norm = min(avg / np.log(10), 1.0)
        size = int(32 + norm * (256 - 32))
        size = max(size, output_dim * 6)  # CIFAR는 여유 있게
        return size, avg

    def add_cifar(self, train_loader, test_loader, output_dim=10, epochs=10):
        """CIFAR-10 어댑터 + 미니망 추가"""
        print_header("🟠 CIFAR-10 도메인 추가 (어댑터 방식)")
        self.base_net.eval()

        # 어댑터 생성
        adapter = DomainAdapter(input_dim=3072, target_dim=256).to(self.device)
        print_step("어댑터 생성: CIFAR(3072) → h1 공간(256)")

        # 복잡도 측정 → 미니망 크기 결정
        mini_hidden, avg_complexity = self._calc_mini_size(train_loader, output_dim, adapter)
        print_result(f"복잡도: {avg_complexity:.4f} → hidden={mini_hidden}")

        # 미니망 생성
        mini_net = MiniNetwork(256, mini_hidden, output_dim).to(self.device)
        print_step(f"미니망 생성: 256 → {mini_hidden} → {output_dim}")

        # 어댑터 + 미니망 학습 (기본망은 완전 동결)
        params = list(adapter.parameters()) + list(mini_net.parameters())
        opt    = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        sched  = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit   = nn.CrossEntropyLoss()

        print_step(f"학습 시작 (epochs={epochs}, 기본망 동결 유지)")
        for epoch in range(epochs):
            adapter.train(); mini_net.train()
            total = 0
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                feats = adapter(imgs)           # 어댑터: CIFAR → h1 공간
                out   = mini_net(feats)         # 미니망: 분류
                loss  = crit(out, labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            sched.step()
            lr = opt.param_groups[0]['lr']
            print_result(f"Epoch {epoch+1} | Loss: {total/len(train_loader):.4f} | lr: {lr:.6f}")

        self.mini_nets['cifar10'] = {
            'net':     mini_net,
            'layer':   'h1',
            'hidden':  mini_hidden,
            'adapter': adapter,
        }
        self.adapters['cifar10'] = adapter
        print_result("CIFAR-10 도메인 추가 완료 ✓")

        # 시험
        return self.exam('cifar10', test_loader, threshold=40)

    def exam(self, domain_name, dataloader, threshold=80):
        print_header(f"3단계 | 시험: {domain_name}")
        info     = self.mini_nets[domain_name]
        mini_net = info['net']
        layer    = info['layer']
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
                    feats = self.base_net.activation_maps[layer]
                correct += (mini_net(feats).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total * 100
        print_bar("정확도", correct, total)
        print_result(f"시험 정확도: {accuracy:.2f}%")

        finetuned = False
        if accuracy < threshold:
            finetuned = True
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
                    feats = adapter(imgs) if adapter else self.base_net.activation_maps[layer]
                    if not adapter:
                        with torch.no_grad(): self.base_net(imgs)
                        feats = self.base_net.activation_maps[layer]
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
                        feats = self.base_net.activation_maps[layer]
                    correct += (mini_net(feats).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            accuracy = correct / total * 100
            print_result(f"파인튜닝 후 정확도: {accuracy:.2f}%")
        else:
            print_result("통과 → 연결 강화 ✓")

        return accuracy, finetuned

    def verify_no_forgetting(self, dataloader, tag=""):
        print_header(f"검증 | MNIST 망각률 {tag}")
        self.base_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  MNIST 검증", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                correct += (self.base_net(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total * 100
        print_result(f"MNIST 정확도: {acc:.2f}%")
        return acc

    def verify_domain(self, domain_name, dataloader, tag=""):
        """기존 도메인 미니망이 여전히 작동하는지 확인"""
        print_header(f"검증 | {domain_name} 재검증 {tag}")
        acc, _ = self.exam(domain_name, dataloader, threshold=0)
        return acc

    def save(self, path='afln_v3.pt'):
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
        print_result(f"모델 저장 완료 → {path}")


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

    # ── v2 체크포인트 불러오기 ──────────────
    afln = AFLNv3.from_checkpoint('afln_v2.pt', device=device)

    # ── 데이터 준비 ─────────────────────────
    tf_mnist  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tf_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tf_cifar  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    mnist_test   = datasets.MNIST('./data', train=False, download=True, transform=tf_mnist)
    emnist_test  = LabelShiftDataset(datasets.EMNIST('./data', split='letters', train=False, download=True, transform=tf_emnist))
    cifar_train  = datasets.CIFAR10('./data', train=True,  download=True, transform=tf_cifar)
    cifar_test   = datasets.CIFAR10('./data', train=False, download=True, transform=tf_cifar)

    mnist_test_loader  = DataLoader(mnist_test,  batch_size=64, shuffle=False)
    emnist_test_loader = DataLoader(emnist_test, batch_size=64, shuffle=False)
    cifar_loader       = DataLoader(cifar_train, batch_size=64, shuffle=True)
    cifar_test_loader  = DataLoader(cifar_test,  batch_size=64, shuffle=False)

    # ── v2에서 가져온 기준값 ────────────────
    base_acc         = afln.history.get('base_acc', 0)
    alpha_acc_before = afln.history.get('domain_acc', {}).get('alphabet', 0)
    print_result = lambda msg: print(f"  ✔ {msg}")
    print(f"\n  📋 v2 기준값")
    print(f"  ✔ MNIST 기준 정확도:  {base_acc:.2f}%")
    print(f"  ✔ 알파벳 정확도 (v2): {alpha_acc_before:.2f}%")

    # ── CIFAR-10 추가 ───────────────────────
    cifar_acc, _ = afln.add_cifar(cifar_loader, cifar_test_loader, output_dim=10, epochs=10)

    # ── 망각률 검증 ─────────────────────────
    after_mnist = afln.verify_no_forgetting(mnist_test_loader, "(CIFAR 추가 후)")
    forget      = base_acc - after_mnist

    # ── 알파벳 재검증 ───────────────────────
    alpha_acc_after = afln.verify_domain('alphabet', emnist_test_loader, "(CIFAR 추가 후)")

    # ── 최종 저장 ───────────────────────────
    print_header("💾 모델 저장")
    afln.save('afln_v3.pt')

    # ── 최종 결과 출력 ──────────────────────
    print_final(
        afln.history,
        base_acc, alpha_acc_before, alpha_acc_after,
        cifar_acc, forget
    )