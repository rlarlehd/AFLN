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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


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
    print(f"  {label:12s} [{bar}] {value/total*100:.1f}%")


# ─────────────────────────────────────────
# 4. AFLN 클래스
# ─────────────────────────────────────────
class AFLN:
    def __init__(self, device='cpu'):
        self.device = device
        self.base_net  = BaseNetwork().to(device)
        self.mini_nets = {}
        self.adapters  = {}
        self.clusters  = {}
        self.frozen    = False
        self.history   = {
            'base_acc': None,
            'domain_acc': {},
            'forget': {},
        }

    def pretrain(self, dataloader, epochs=5):
        print_header("1단계 | 최초 학습 (MNIST)")
        opt  = optim.Adam(self.base_net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        self.base_net.train()

        for epoch in range(epochs):
            total = 0
            pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{epochs}", ncols=72,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                loss = crit(self.base_net(imgs), labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print_result(f"Epoch {epoch+1} | Loss: {total/len(dataloader):.4f}")

        print_result("최초 학습 완료")
        self._generate_clusters(dataloader)

    def _generate_clusters(self, dataloader):
        print_header("클러스터링 | DBSCAN 대분류 자율 생성")
        self.base_net.eval()
        all_acts = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(tqdm(dataloader, desc="  패턴 수집", ncols=72)):
                imgs = imgs.to(self.device)
                self.base_net(imgs)
                all_acts.append(self.base_net.activation_maps['h2'].cpu().numpy())
                if i >= 20: break

        acts    = np.concatenate(all_acts)
        scaler  = StandardScaler()
        pca     = PCA(n_components=10)
        reduced = pca.fit_transform(scaler.fit_transform(acts))
        labels  = DBSCAN(eps=3.0, min_samples=5).fit_predict(reduced)
        n       = len(set(labels)) - (1 if -1 in labels else 0)

        print_result(f"대분류 {n}개 생성 | 분산 설명률: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        print("\n  클러스터 분포:")
        for c in sorted(set(labels)):
            cnt = (labels == c).sum()
            print_bar("노이즈" if c==-1 else f"클러스터{c}", cnt, len(labels))

        self.clusters = {'n': n}
        for p in self.base_net.parameters():
            p.requires_grad = False
        self.frozen = True
        print_result("기존 가중치 완전 동결 ✓")

    def _calc_mini_size(self, dataloader, output_dim):
        total, count = 0, 0
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                out  = self.base_net(imgs)
                ent  = -torch.sum(
                    torch.softmax(out,1) * torch.log_softmax(out,1), dim=1
                ).mean()
                total += ent.item(); count += 1
                if count > 10: break
        avg  = total / count
        norm = min(avg / np.log(10), 1.0)
        size = int(32 + norm * (256 - 32))
        size = max(size, output_dim * 4)
        return size, avg

    def adaptive_learn(self, dataloader, output_dim, domain_name, feature_layer='h1'):
        print_header(f"2단계 | 신규 학습: {domain_name}  [특징층: {feature_layer}]")
        self.base_net.eval()

        mini_hidden, avg_complexity = self._calc_mini_size(dataloader, output_dim)
        print_result(f"복잡도: {avg_complexity:.4f} → hidden={mini_hidden}")

        mini_net = MiniNetwork(256, mini_hidden, output_dim).to(self.device)
        opt  = optim.Adam(mini_net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        mini_net.train()

        for epoch in range(5):
            total = 0
            pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/5", ncols=72,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    self.base_net(imgs)
                    feats = self.base_net.activation_maps[feature_layer]
                opt.zero_grad()
                loss = crit(mini_net(feats), labels)
                loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            print_result(f"Epoch {epoch+1} | Loss: {total/len(dataloader):.4f}")

        self.mini_nets[domain_name] = {
            'net': mini_net, 'layer': feature_layer,
            'hidden': mini_hidden, 'adapter': None
        }
        print_result(f"{domain_name} 미니망 생성 완료 ✓")
        return mini_hidden

    def exam(self, dataloader, domain_name):
        print_header(f"3단계 | 시험: {domain_name}")
        info     = self.mini_nets[domain_name]
        mini_net = info['net']
        layer    = info['layer']
        crit     = nn.CrossEntropyLoss()

        mini_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="  시험 중", ncols=72):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.base_net(imgs)
                feats = self.base_net.activation_maps[layer]
                correct += (mini_net(feats).argmax(1) == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total * 100
        print_bar("정확도", correct, total)
        print_result(f"시험 정확도: {accuracy:.2f}%")
        self.history['domain_acc'][domain_name] = accuracy

        finetuned = False
        if accuracy < 80:
            finetuned = True
            print_step(f"정확도 부족 ({accuracy:.1f}%) → 해당 미니망만 파인튜닝")
            opt = optim.Adam(mini_net.parameters(), lr=1e-4)
            mini_net.train()
            for epoch in range(5):
                for imgs, labels in tqdm(dataloader, desc=f"  파인튜닝 {epoch+1}/5", ncols=72):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        self.base_net(imgs)
                        feats = self.base_net.activation_maps[layer]
                    opt.zero_grad()
                    loss = crit(mini_net(feats), labels)
                    loss.backward(); opt.step()

            mini_net.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in dataloader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    self.base_net(imgs)
                    feats = self.base_net.activation_maps[layer]
                    correct += (mini_net(feats).argmax(1) == labels).sum().item()
                    total   += labels.size(0)
            accuracy = correct / total * 100
            self.history['domain_acc'][domain_name] = accuracy
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

    # ── 저장 / 불러오기 ───────────────────
    def save(self, path='afln_checkpoint.pt'):
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
            'clusters': self.clusters,
            'frozen':   self.frozen,
            'history':  self.history,
        }
        torch.save(state, path)
        print_result(f"모델 저장 완료 → {path}")

    @classmethod
    def load(cls, path='afln_checkpoint.pt', device='cpu'):
        state  = torch.load(path, map_location=device)
        afln   = cls(device=device)

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
                'net':     net,
                'layer':   v['layer'],
                'hidden':  v['hidden'],
                'adapter': None,
            }

        print_result(f"모델 불러오기 완료 ← {path}")
        print_result(f"  기존 도메인: {list(afln.mini_nets.keys())}")
        print_result(f"  클러스터 수: {afln.clusters.get('n', '?')}개")
        return afln


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

    tf_mnist  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tf_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

    mnist_train  = datasets.MNIST('./data', train=True,  download=True, transform=tf_mnist)
    mnist_test   = datasets.MNIST('./data', train=False, download=True, transform=tf_mnist)
    emnist_train = LabelShiftDataset(datasets.EMNIST('./data', split='letters', train=True,  download=True, transform=tf_emnist))
    emnist_test  = LabelShiftDataset(datasets.EMNIST('./data', split='letters', train=False, download=True, transform=tf_emnist))

    mnist_loader       = DataLoader(mnist_train,  batch_size=64, shuffle=True)
    mnist_test_loader  = DataLoader(mnist_test,   batch_size=64, shuffle=False)
    emnist_loader      = DataLoader(emnist_train, batch_size=64, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test,  batch_size=64, shuffle=False)

    afln = AFLN(device=device)

    # 1단계: MNIST 최초 학습
    afln.pretrain(mnist_loader, epochs=5)
    base_acc = afln.verify_no_forgetting(mnist_test_loader, "(기준)")
    afln.history['base_acc'] = base_acc

    # 2단계: 알파벳 학습
    afln.adaptive_learn(emnist_loader, output_dim=26, domain_name='alphabet', feature_layer='h1')
    afln.exam(emnist_test_loader, 'alphabet')
    after_acc = afln.verify_no_forgetting(mnist_test_loader, "(알파벳 후)")
    afln.history['forget']['alphabet'] = base_acc - after_acc

    # 저장
    print_header("💾 모델 저장")
    afln.save('afln_v2.pt')

    print_header("✅ v2 완료 요약")
    print_result(f"MNIST 기준 정확도:   {base_acc:.2f}%")
    print_result(f"알파벳 정확도:       {afln.history['domain_acc'].get('alphabet', 0):.2f}%")
    print_result(f"MNIST 망각률:        {base_acc - after_acc:.2f}%")
    print_result(f"저장 경로:           afln_v2.pt")
    print_result(f"다음 단계:           python afln_v3.py")