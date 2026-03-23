import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
# 1. 라우터
#    - 입력을 보고 도메인 번호 출력
#    - 새 도메인 추가 시 출력 노드 1개 증가
# ─────────────────────────────────────────
class Router(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = None  # 도메인 수에 따라 동적 생성
        self.n_domains = 0

    def add_domain(self, device):
        """새 도메인 추가 시 출력 노드 1개 증가"""
        self.n_domains += 1
        old_head = self.head
        new_head  = nn.Linear(64, self.n_domains).to(device)

        # 기존 가중치 보존
        if old_head is not None:
            with torch.no_grad():
                new_head.weight[:old_head.out_features] = old_head.weight
                new_head.bias[:old_head.out_features]   = old_head.bias
        self.head = new_head

    def forward(self, x):
        x   = x.view(x.size(0), -1)
        feat = self.net(x)
        return self.head(feat)

    def predict_domain(self, x):
        """도메인 번호 예측"""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)


# ─────────────────────────────────────────
# 2. 동적 공유망
#    - 노드 소유권 관리
#    - 새 도메인 시 새 노드 생성
#    - forward는 항상 전체 노드 통과
# ─────────────────────────────────────────
class DynamicLayer(nn.Module):
    def __init__(self, in_features, init_nodes):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(init_nodes, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(init_nodes))
        self.node_owner   = {}
        self.domain_nodes = {}

    @property
    def out_features(self):
        return self.weight.shape[0]

    def expand_input(self, n_new, device):
        new_cols = torch.zeros(self.weight.shape[0], n_new).to(device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_cols], dim=1))
        self.in_features += n_new

    def add_nodes(self, n, domain_name, device):
        old = self.weight.shape[0]
        new_w = torch.randn(n, self.in_features).to(device) * 0.01
        new_b = torch.zeros(n).to(device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
        self.bias   = nn.Parameter(torch.cat([self.bias.data,   new_b], dim=0))
        indices = list(range(old, old + n))
        self._register(domain_name, indices)
        return indices

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
        owned = set(self.domain_nodes.get(domain_name, []))
        for param, is_row in [(self.weight, True), (self.bias, False)]:
            if param.grad is None: continue
            mask = torch.zeros_like(param.grad)
            for idx in owned:
                if is_row: mask[idx] = 1.0
                else:      mask[idx] = 1.0
            param.grad *= mask

    def forward(self, x):
        return torch.relu(x @ self.weight.T + self.bias)


class SharedNetwork(nn.Module):
    def __init__(self, input_dim=784, init_nodes=256):
        super().__init__()
        self.layer1 = DynamicLayer(input_dim,   init_nodes)
        self.layer2 = DynamicLayer(init_nodes,  init_nodes)
        self.heads  = nn.ModuleDict()
        self.head_nodes     = {}
        self.activation_maps = {}

    def add_head(self, domain_name, output_dim, node_indices):
        device = self.layer1.weight.device
        self.heads[domain_name]      = nn.Linear(len(node_indices), output_dim).to(device)
        self.head_nodes[domain_name] = node_indices

    def forward(self, x, domain=None):
        x  = x.view(x.size(0), -1)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        self.activation_maps['h1'] = h1.detach()
        self.activation_maps['h2'] = h2.detach()
        if domain and domain in self.heads:
            idx  = self.head_nodes[domain]
            return self.heads[domain](h2[:, idx])
        # 헤드 없으면 h2만 반환 (활성화 탐지용)
        if not self.heads:
            return h2
        first = list(self.heads.keys())[0]
        idx   = self.head_nodes[first]
        return self.heads[first](h2[:, idx])


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
# 4. AFLN v9 메인
# ─────────────────────────────────────────
class AFLNv9:
    def __init__(self, device='cpu', init_nodes=256):
        self.device      = device
        self.init_nodes  = init_nodes
        self.router      = Router().to(device)
        self.net         = SharedNetwork(init_nodes=init_nodes).to(device)
        self.domain_list = []   # 등록 순서 [domain_name, ...]
        self.history     = {'domain_acc': {}, 'router_acc': {}, 'forget': {}}

    # ── 도메인 등록 및 학습 ───────────────
    def register_domain(self, domain_name, train_loader,
                        output_dim, epochs=7, min_nodes=64):
        domain_id = len(self.domain_list)
        self.domain_list.append(domain_name)
        print_header(f"도메인 {domain_id} 등록: {domain_name}")

        # 1. 라우터 출력 노드 추가
        self.router.add_domain(self.device)
        print_result(f"라우터 확장: {domain_id+1}개 도메인 인식 가능")

        # 2. 공유망 노드 할당
        self.net.eval()
        active_l2 = self._detect_active_nodes(train_loader)
        free_l2   = self.net.layer2.get_free_nodes(active_l2)
        need      = max(0, min_nodes - len(free_l2))

        new_l2 = []
        if need > 0:
            new_l2 = self.net.layer2.add_nodes(need, domain_name, self.device)
            print_result(f"새 노드 생성: L2 +{need}개 (총 {self.net.layer2.out_features}개)")

        # layer1도 동일하게
        active_l1 = self._detect_active_nodes_l1(train_loader)
        free_l1   = self.net.layer1.get_free_nodes(active_l1)
        need_l1   = max(0, min_nodes - len(free_l1))
        new_l1    = []
        if need_l1 > 0:
            new_l1 = self.net.layer1.add_nodes(need_l1, domain_name, self.device)
            self.net.layer2.expand_input(need_l1, self.device)
            print_result(f"새 노드 생성: L1 +{need_l1}개 (총 {self.net.layer1.out_features}개)")

        domain_l1 = free_l1[:min_nodes] + new_l1
        domain_l2 = free_l2[:min_nodes] + new_l2
        self.net.layer1.assign_nodes(domain_l1, domain_name)
        self.net.layer2.assign_nodes(domain_l2, domain_name)
        self.net.add_head(domain_name, output_dim, domain_l2)
        print_result(f"소유권 등록: L1={len(domain_l1)}개, L2={len(domain_l2)}개")

        # 3. 공유망 학습 (소유 노드 + 헤드만)
        self._train_network(domain_name, train_loader, epochs)

        # 4. 라우터 학습 (모든 기존 도메인 데이터로)
        self._train_router(train_loader, domain_id)

    # ── 활성화 노드 탐지 ──────────────────
    def _detect_active_nodes(self, dataloader, k=128):
        all_h2 = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                self.net(imgs)
                all_h2.append(self.net.activation_maps['h2'].cpu().numpy())
                if i >= 10: break
        mean = np.concatenate(all_h2).mean(axis=0)
        return np.argsort(mean)[-k:].tolist()

    def _detect_active_nodes_l1(self, dataloader, k=128):
        all_h1 = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                self.net(imgs)
                all_h1.append(self.net.activation_maps['h1'].cpu().numpy())
                if i >= 10: break
        mean = np.concatenate(all_h1).mean(axis=0)
        return np.argsort(mean)[-k:].tolist()

    # ── 공유망 학습 ───────────────────────
    def _train_network(self, domain_name, train_loader, epochs):
        print_step(f"공유망 학습: {domain_name} 소유 노드만 업데이트")
        opt  = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
        crit = nn.CrossEntropyLoss()

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

    # ── 라우터 학습 ───────────────────────
    def _train_router(self, new_loader, new_domain_id, epochs=3):
        """라우터: 새 도메인 추가 시 전체 도메인 구분 재학습"""
        print_step(f"라우터 학습: {len(self.domain_list)}개 도메인 구분")

        # 도메인 레이블 데이터셋 생성
        # 새 도메인 데이터에 domain_id 레이블 붙이기
        class DomainLabelDataset(Dataset):
            def __init__(self, ds, domain_id):
                self.ds        = ds
                self.domain_id = domain_id
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                img, _ = self.ds[i]
                return img, self.domain_id

        # 각 도메인 데이터 수집 (저장된 로더에서)
        combined = DomainLabelDataset(new_loader.dataset, new_domain_id)
        router_loader = DataLoader(combined, batch_size=64, shuffle=True)

        opt  = optim.Adam(self.router.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.router.train()
            total = 0
            pbar  = tqdm(router_loader, desc=f"  라우터 Epoch {epoch+1}/{epochs}", ncols=72,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, d_labels in pbar:
                imgs, d_labels = imgs.to(self.device), d_labels.to(self.device)
                opt.zero_grad()
                out  = self.router(imgs)
                loss = crit(out, d_labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print_result(f"라우터 Epoch {epoch+1} | Loss: {total/len(router_loader):.4f}")

    # ── 추론 (라우터 → 공유망) ─────────────
    def predict(self, imgs):
        """라우터가 도메인 판단 → 해당 헤드로 라우팅"""
        imgs = imgs.to(self.device)
        domain_ids = self.router.predict_domain(imgs)

        results = torch.zeros(imgs.size(0), dtype=torch.long).to(self.device)
        for d_id in domain_ids.unique():
            mask        = (domain_ids == d_id)
            domain_name = self.domain_list[d_id.item()]
            out         = self.net(imgs[mask], domain=domain_name)
            results[mask] = out.argmax(dim=1)
        return results, domain_ids

    # ── 시험 ─────────────────────────────
    def exam(self, domain_name, dataloader, threshold=80):
        print_header(f"시험: {domain_name}")
        domain_id = self.domain_list.index(domain_name)
        self.net.eval(); self.router.eval()
        correct_cls, correct_route, total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  시험 중", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds, d_ids = self.predict(imgs)

                # 분류 정확도 (도메인 맞게 라우팅된 것만)
                routed_ok = (d_ids == domain_id)
                if routed_ok.any():
                    correct_cls += (preds[routed_ok] == labels[routed_ok]).sum().item()
                correct_route += routed_ok.sum().item()
                total         += labels.size(0)

        route_acc = correct_route / total * 100
        cls_acc   = correct_cls / max(correct_route, 1) * 100
        overall   = correct_cls / total * 100

        print_bar("라우팅 정확도", correct_route, total)
        print_bar("분류 정확도",   correct_cls,   max(correct_route, 1))
        print_result(f"종합 정확도: {overall:.2f}%  "
                     f"(라우팅 {route_acc:.1f}% × 분류 {cls_acc:.1f}%)")

        self.history['domain_acc'][domain_name]  = overall
        self.history['router_acc'][domain_name]  = route_acc
        return overall

    # ── 전체 현황 ─────────────────────────
    def print_snapshot(self, base_acc):
        print_header("📊 전체 현황 (라우터 + 공유망)")
        print(f"  {'도메인':<16} {'종합':>8} {'라우팅':>8} {'망각률':>8} {'판정':>6}")
        print(f"  {'─'*50}")
        for dn, acc in self.history['domain_acc'].items():
            forget  = base_acc - acc if dn == self.domain_list[0] \
                      else self.history['forget'].get(dn, 0)
            route   = self.history['router_acc'].get(dn, 0)
            judge   = '✔' if forget < 2.0 else '△' if forget < 5.0 else '✘'
            print(f"  {dn:<16} {acc:>7.2f}% {route:>7.1f}% {forget:>7.2f}% {judge:>6}")
        print(f"  {'─'*50}")
        print(f"\n  네트워크 노드 현황:")
        print(f"  Layer1 총 {self.net.layer1.out_features}개 | "
              f"Layer2 총 {self.net.layer2.out_features}개")
        for dn in self.net.layer1.domain_nodes:
            l1 = len(self.net.layer1.domain_nodes.get(dn, []))
            l2 = len(self.net.layer2.domain_nodes.get(dn, []))
            print(f"  {dn:<16} L1: {l1:3}개 | L2: {l2:3}개")

    # ── 저장 ─────────────────────────────
    def save(self, path='afln_v9.pt'):
        torch.save({
            'router':        self.router.state_dict(),
            'net':           self.net.state_dict(),
            'domain_list':   self.domain_list,
            'head_nodes':    self.net.head_nodes,
            'layer1_owner':  self.net.layer1.node_owner,
            'layer1_domain': self.net.layer1.domain_nodes,
            'layer2_owner':  self.net.layer2.node_owner,
            'layer2_domain': self.net.layer2.domain_nodes,
            'history':       self.history,
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
    print(f"  구조: 라우터 → 공유망 → 도메인별 헤드")

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

    afln = AFLNv9(device=device, init_nodes=256)

    # 도메인 0: MNIST
    afln.register_domain('mnist',    mnist_loader,  output_dim=10, epochs=5, min_nodes=128)
    base_acc = afln.exam('mnist', mnist_test_loader)
    afln.history['domain_acc']['mnist'] = base_acc

    # 도메인 1: 알파벳
    afln.register_domain('alphabet', emnist_loader, output_dim=26, epochs=7, min_nodes=80)
    afln.exam('alphabet', emnist_test_loader, threshold=80)
    after_alpha = afln.exam('mnist', mnist_test_loader)
    afln.history['forget']['alphabet'] = base_acc - after_alpha
    afln.print_snapshot(base_acc)

    # 도메인 2: Fashion
    afln.register_domain('fashion',  fmnist_loader, output_dim=10, epochs=7, min_nodes=64)
    afln.exam('fashion', fmnist_test_loader, threshold=70)
    after_fashion = afln.exam('mnist', mnist_test_loader)
    afln.history['forget']['fashion'] = base_acc - after_fashion

    # 알파벳 재검증
    print_header("🔍 알파벳 재검증 (Fashion 추가 후)")
    after_alpha2 = afln.exam('alphabet', emnist_test_loader, threshold=0)
    afln.print_snapshot(base_acc)

    # 저장
    afln.save('afln_v9.pt')

    # 최종 비교
    print_header("🏁 최종 비교")
    print(f"  {'항목':<24} {'v6 EWC':>9} {'v9 라우터':>10}")
    print(f"  {'─'*46}")
    print(f"  {'MNIST 망각률':<24} {'4.82%':>9} {base_acc-after_fashion:>9.2f}%")
    print(f"  {'알파벳 정확도':<24} {'72.58%':>9} {after_alpha2:>9.2f}%")
    print(f"  {'Fashion 정확도':<24} {'85.27%':>9} "
          f"{afln.history['domain_acc'].get('fashion',0):>9.2f}%")
    print(f"  {'─'*46}")
    print(f"\n  🆕 v9 추가 지표:")
    for dn in afln.domain_list:
        ra = afln.history['router_acc'].get(dn, 0)
        print(f"  라우팅 정확도 ({dn}): {ra:.1f}%")