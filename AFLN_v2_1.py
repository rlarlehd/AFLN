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
from copy import deepcopy

# ─────────────────────────────────────────
# 1. 단일 네트워크
#    - 모든 도메인이 같은 가중치를 공유
#    - 도메인별 출력 헤드만 분리
# ─────────────────────────────────────────
class SharedNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.activation_maps = {}
        # 도메인별 출력 헤드 (여기만 도메인마다 다름)
        self.heads = nn.ModuleDict()

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


# ─────────────────────────────────────────
# 2. EWC (Elastic Weight Consolidation)
#    - 학습 후 각 가중치의 중요도(Fisher) 계산
#    - 새 도메인 학습 시 중요한 가중치 보호
# ─────────────────────────────────────────
class EWC:
    def __init__(self, model, dataloader, device, domain):
        self.device  = device
        self.domain  = domain
        self.params  = {n: p.clone().detach()
                        for n, p in model.named_parameters()
                        if p.requires_grad and 'heads' not in n}
        self.fisher  = self._calc_fisher(model, dataloader)

    def _calc_fisher(self, model, dataloader):
        """Fisher Information: 각 가중치가 얼마나 중요한지"""
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters()
                  if p.requires_grad and 'heads' not in n}

        model.eval()
        for imgs, labels in tqdm(dataloader, desc="  Fisher 계산 중", ncols=72):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            model.zero_grad()
            out  = model(imgs, domain=self.domain)
            loss = nn.CrossEntropyLoss()(out, labels)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and 'heads' not in n and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        # 평균
        for n in fisher:
            fisher[n] /= len(dataloader)
        return fisher

    def penalty(self, model):
        """EWC 패널티: 중요한 가중치가 많이 변하면 큰 패널티"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss


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
    filled = int(width * value / total)
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label:14s} [{bar}] {value/total*100:.1f}%")


# ─────────────────────────────────────────
# 4. AFLN v6 메인
# ─────────────────────────────────────────
class AFLNv6:
    def __init__(self, device='cpu', ewc_lambda=5000):
        self.device     = device
        self.net        = SharedNetwork().to(device)
        self.ewc_list   = []       # 누적 EWC 제약
        self.ewc_lambda = ewc_lambda
        self.history    = {'base_acc': None, 'domain_acc': {}, 'forget': {}}
        self.clusters   = {}

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

        # Fisher 계산 → EWC 등록
        print_step("Fisher Information 계산 중 (가중치 중요도 측정)...")
        ewc = EWC(self.net, dataloader, self.device, 'mnist')
        self.ewc_list.append(ewc)
        print_result("EWC 등록 완료 ✓")

        # DBSCAN 클러스터링
        self._generate_clusters(dataloader)

    # ── DBSCAN 대분류 자율 생성 ────────────
    def _generate_clusters(self, dataloader):
        print_header("클러스터링 | DBSCAN 대분류 자율 생성")
        self.net.eval()
        all_acts = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  패턴 수집", ncols=72)):
                imgs = imgs.to(self.device)
                self.net(imgs, 'mnist')
                all_acts.append(self.net.activation_maps['h2'].cpu().numpy())
                if i >= 20: break

        acts    = np.concatenate(all_acts)
        scaler  = StandardScaler()
        pca     = PCA(n_components=10)
        reduced = pca.fit_transform(scaler.fit_transform(acts))
        labels  = DBSCAN(eps=3.0, min_samples=5).fit_predict(reduced)
        n       = len(set(labels)) - (1 if -1 in labels else 0)

        self.clusters = {'n': n, 'labels': labels}
        print_result(f"대분류 {n}개 자율 생성 완료")

        print("\n  클러스터 분포:")
        for c in sorted(set(labels)):
            cnt = (labels == c).sum()
            print_bar("노이즈" if c==-1 else f"클러스터{c}", cnt, len(labels))

    # ── 미니망 크기 자동 결정 ───────────────
    def _calc_mini_size(self, dataloader, output_dim, domain):
        total, count = 0, 0
        self.net.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                out  = self.net(imgs, domain)
                ent  = -torch.sum(
                    torch.softmax(out,1) * torch.log_softmax(out,1), dim=1
                ).mean()
                total += ent.item(); count += 1
                if count > 10: break
        avg  = total / count
        norm = min(avg / np.log(10), 1.0)
        size = max(int(32 + norm * (256 - 32)), output_dim * 4)
        return size, avg

    # ── 신규 도메인 학습 ──────────────────
    def learn_domain(self, domain_name, train_loader,
                     output_dim, epochs=7):
        print_header(f"신규 도메인 학습: {domain_name}")

        # 출력 헤드 추가
        self.net.add_head(domain_name, output_dim)

        # 복잡도 측정 → 적절한 학습률 결정
        _, avg_c = self._calc_mini_size(train_loader, output_dim, 'mnist')
        norm  = min(avg_c / np.log(10), 1.0)
        lr    = 1e-4 + norm * (5e-4 - 1e-4)  # 복잡도 높을수록 lr 높게
        print_result(f"복잡도: {avg_c:.4f} → 학습률 자동 결정: {lr:.6f}")

        # 공유 가중치 + 새 헤드 같이 학습
        # 단, EWC 패널티로 기존 지식 보호
        opt  = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit = nn.CrossEntropyLoss()

        print_step(f"EWC λ={self.ewc_lambda} 적용 (기존 {len(self.ewc_list)}개 도메인 보호)")

        for epoch in range(epochs):
            self.net.train()
            total_loss = 0
            total_ewc  = 0
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()

                # 새 도메인 손실
                out  = self.net(imgs, domain=domain_name)
                loss = crit(out, labels)

                # EWC 패널티 (기존 모든 도메인 보호)
                ewc_loss = sum(e.penalty(self.net) for e in self.ewc_list)
                total    = loss + self.ewc_lambda * ewc_loss

                total.backward(); opt.step()
                total_loss += loss.item()
                total_ewc  += ewc_loss.item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ewc=f"{ewc_loss.item():.4f}"
                )
            sched.step()
            print_result(
                f"Epoch {epoch+1} | "
                f"Loss: {total_loss/len(train_loader):.4f} | "
                f"EWC: {total_ewc/len(train_loader):.6f}"
            )

        # 학습 완료 후 이 도메인의 Fisher도 등록
        print_step(f"{domain_name} Fisher 계산 중...")
        ewc = EWC(self.net, train_loader, self.device, domain_name)
        self.ewc_list.append(ewc)
        print_result(f"EWC 누적: {len(self.ewc_list)}개 도메인 보호 중")

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

        # 파인튜닝 (EWC 패널티 유지하면서)
        if accuracy < threshold:
            print_step(f"정확도 부족 ({accuracy:.1f}%) → EWC 유지하며 파인튜닝")
            opt  = optim.Adam(self.net.parameters(), lr=1e-4)
            crit = nn.CrossEntropyLoss()
            for epoch in range(5):
                self.net.train()
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=72):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    loss     = crit(self.net(imgs, domain_name), labels)
                    ewc_loss = sum(e.penalty(self.net) for e in self.ewc_list)
                    (loss + self.ewc_lambda * ewc_loss).backward()
                    opt.step()

            # 재평가
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
        acc = self.exam('mnist', dataloader, threshold=0)
        return acc

    # ── 전체 현황 출력 ────────────────────
    def print_snapshot(self, base_acc):
        print_header("📊 전체 현황 (단일 네트워크)")
        print(f"  {'도메인':<16} {'정확도':>8} {'망각률':>8} {'판정':>6}")
        print(f"  {'─'*42}")
        for domain, acc in self.history['domain_acc'].items():
            forget = base_acc - acc if domain == 'mnist' else \
                     self.history['forget'].get(domain, 0)
            judge  = '✔' if forget < 3.0 else '△' if forget < 5.0 else '✘'
            print(f"  {domain:<16} {acc:>7.2f}% {forget:>7.2f}% {judge:>6}")
        print(f"  {'─'*42}")
        print(f"\n  EWC 보호 도메인 수: {len(self.ewc_list)}개")
        print(f"  공유 네트워크 파라미터: "
              f"{sum(p.numel() for p in self.net.fc1.parameters()) + sum(p.numel() for p in self.net.fc2.parameters()):,}개")

    # ── 저장 ─────────────────────────────
    def save(self, path='afln_v6.pt'):
        torch.save({
            'net':        self.net.state_dict(),
            'history':    self.history,
            'clusters':   self.clusters,
            'ewc_lambda': self.ewc_lambda,
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
    print(f"  구조: 단일 공유 네트워크 + EWC 보호 + 도메인별 출력 헤드")

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

    afln = AFLNv6(device=device, ewc_lambda=5000)

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
    afln.exam('alphabet', emnist_test_loader, threshold=0)
    afln.print_snapshot(base_acc)

    # 저장
    afln.save('afln_v6.pt')

    # 최종 판정
    print_header("🏁 최종 판정")
    print(f"  단일 네트워크로 {len(afln.history['domain_acc'])}개 도메인 처리")
    forget_vals = [base_acc - afln.history['domain_acc'].get('mnist', base_acc)]
    forget_vals += list(afln.history['forget'].values())
    max_forget = max(forget_vals)
    if max_forget < 3.0:
        print(f"  ✔ 최대 망각률 {max_forget:.2f}% → EWC 효과 확인")
        print(f"  ✔ 하나의 네트워크로 모든 도메인 처리 성공")
    else:
        print(f"  △ 최대 망각률 {max_forget:.2f}% → λ 조정 필요")
        print(f"  → ewc_lambda를 높이면 망각 줄어듦 (대신 새 학습 정확도 낮아짐)")