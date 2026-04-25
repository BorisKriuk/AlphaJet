import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import voxelize_jet, sample_params
from advae import ADVAE, advae_loss, normalize_params


class JetDataset(Dataset):
    def __init__(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n):
            p = sample_params(rng)
            self.samples.append((voxelize_jet(p), normalize_params(p)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        g, p = self.samples[i]
        return torch.from_numpy(g).unsqueeze(0), p


def train(n_samples=4000, epochs=40, batch_size=32,
          device='cpu', save_path='advae.pt'):
    print(f"[train] generating {n_samples} synthetic jets...")
    loader = DataLoader(JetDataset(n=n_samples), batch_size=batch_size,
                        shuffle=True, num_workers=0)

    model = ADVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        tot = rec = kl = an = 0.0
        # KL warm-up so decoder learns shapes first, then we regularize latent
        beta = min(1.0, ep / 10.0) * 0.5
        for x, p in loader:
            x, p = x.to(device), p.to(device)
            recon, mu, logvar, _ = model(x)
            loss, r, k, a = advae_loss(recon, x, mu, logvar, p,
                                       beta=beta, lambda_anat=30.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item(); rec += r.item(); kl += k.item(); an += a.item()
        sched.step()
        n = len(loader)
        print(f"[train] ep {ep+1:02d}/{epochs}  "
              f"loss={tot/n:.3f}  recon={rec/n:.3f}  "
              f"kl={kl/n:.3f}  anat={an/n:.4f}  beta={beta:.2f}")

    torch.save(model.state_dict(), save_path)
    print(f"[train] saved -> {save_path}")
    return model


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(device=dev)