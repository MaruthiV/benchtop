import math
import numpy as np
import torch
import torch.nn as nn

G = 8
DT_D = 0.2          # true diffusion
NOISE = 0.05        # irreducible observation/process noise
N_TEST = 100
STEPS = 300
BATCH = 64
K_EULER = 5
N_SAMPLES = 4
HID = 64


def _pde_step(x, u, v, D):
    dx = (torch.roll(x, -1, 2) - torch.roll(x, 1, 2)) / 2.0
    dy = (torch.roll(x, -1, 1) - torch.roll(x, 1, 1)) / 2.0
    lap = (torch.roll(x, -1, 2) + torch.roll(x, 1, 2) +
           torch.roll(x, -1, 1) + torch.roll(x, 1, 1) - 4.0 * x)
    return x + (-u * dx - v * dy + D * lap)


_kk = np.fft.fftfreq(G) * G
_K2 = _kk[:, None] ** 2 + _kk[None, :] ** 2
_FILT = np.exp(-_K2 / (2 * 2.0 ** 2))
_FSTD = math.sqrt(np.mean(_FILT ** 2))


def _smooth_fields(rng, n):
    w = rng.standard_normal((n, G, G))
    f = np.fft.ifft2(np.fft.fft2(w) * _FILT).real / _FSTD
    return f


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2 * G * G + 4, HID), nn.SiLU(),
            nn.Linear(HID, HID), nn.SiLU(),
            nn.Linear(HID, G * G))
        self.gate = nn.Parameter(torch.zeros(G * G))

    def forward(self, z, x, tau, p, physics):
        te = torch.cat([torch.sin(math.pi * tau), torch.cos(math.pi * tau),
                        torch.sin(2 * math.pi * tau), torch.cos(2 * math.pi * tau)], 1)
        out = self.f(torch.cat([z, x, te], 1))
        if not physics:
            return out
        g = torch.sigmoid(self.gate)[None]
        phys = (p - z) / torch.clamp(1.0 - tau, min=1.0 / K_EULER)
        return g * out + (1.0 - g) * phys


def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    method = config.get("method", "pgfm")
    n_train = int(config.get("n_train", 200))
    pde = config.get("pde", "exact")
    h = int(config.get("horizon", 1))
    physics = (method == "pgfm")

    # ---------- data (own rng; independent of method) ----------
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 4)
    jj, ii = np.meshgrid(np.arange(G), np.arange(G))
    u_np = 0.2 + 0.25 * np.sin(2 * np.pi * jj / G + ph[0]) * np.cos(2 * np.pi * ii / G + ph[1])
    v_np = 0.15 * np.cos(2 * np.pi * ii / G + ph[2]) * np.sin(2 * np.pi * jj / G + ph[3]) + 0.05
    c_np = 0.4 * _smooth_fields(rng, 1)[0]
    test_x = _smooth_fields(rng, N_TEST)
    test_noise = rng.standard_normal((h, N_TEST, G, G))
    tr_x = _smooth_fields(rng, n_train)
    tr_noise = rng.standard_normal((n_train, G, G))

    u = torch.tensor(u_np, dtype=torch.float32)
    v = torch.tensor(v_np, dtype=torch.float32)
    c = torch.tensor(c_np, dtype=torch.float32)

    if pde == "exact":
        pu, pv, pD = u, v, DT_D
    elif pde == "wind_30":
        pu, pv, pD = 1.3 * u, 1.3 * v, DT_D
    else:  # omit_advection
        pu, pv, pD = torch.zeros_like(u), torch.zeros_like(v), DT_D

    def true_step(x, noise):
        return _pde_step(x, u, v, DT_D) + c - 0.2 * x + 0.15 * torch.sin(2 * x) + NOISE * noise

    def model_pde(x):
        return _pde_step(x, pu, pv, pD)

    X0 = torch.tensor(tr_x, dtype=torch.float32)
    X1 = true_step(X0, torch.tensor(tr_noise, dtype=torch.float32))
    TX = torch.tensor(test_x, dtype=torch.float32)
    truth = []
    cur = TX
    for k in range(h):
        cur = true_step(cur, torch.tensor(test_noise[k], dtype=torch.float32))
        truth.append(cur)

    # ---------- model (separate init seed; same across methods) ----------
    torch.manual_seed(seed + 1000)
    net = _Net()
    gen = torch.Generator().manual_seed(seed + 7)
    if physics:
        opt = torch.optim.Adam([{"params": net.f.parameters(), "lr": 3e-3},
                                {"params": [net.gate], "lr": 3e-2}])
    else:
        opt = torch.optim.Adam(net.f.parameters(), lr=3e-3)

    P0 = model_pde(X0).reshape(n_train, -1)
    F0 = X0.reshape(n_train, -1)
    F1 = X1.reshape(n_train, -1)
    for _ in range(STEPS):
        idx = torch.randint(0, n_train, (BATCH,), generator=gen)
        z0 = torch.randn(BATCH, G * G, generator=gen)
        tau = torch.rand(BATCH, 1, generator=gen)
        z1 = F1[idx]
        z = (1 - tau) * z0 + tau * z1
        pred = net(z, F0[idx], tau, P0[idx], physics)
        loss = ((pred - (z1 - z0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---------- evaluation ----------
    egen = torch.Generator().manual_seed(seed + 99)

    @torch.no_grad()
    def forecast(x):
        n = x.shape[0]
        xf = x.reshape(n, -1).repeat(N_SAMPLES, 1)
        p = model_pde(x).reshape(n, -1).repeat(N_SAMPLES, 1)
        z = torch.randn(xf.shape, generator=egen)
        for k in range(K_EULER):
            tau = torch.full((xf.shape[0], 1), k / K_EULER)
            z = z + net(z, xf, tau, p, physics) / K_EULER
        return z.reshape(N_SAMPLES, n, G, G).mean(0)

    cur = TX
    pde_cur = TX
    with torch.no_grad():
        for k in range(h):
            cur = forecast(cur)
            pde_cur = model_pde(pde_cur)
    tr = truth[-1]
    rmse = float(torch.sqrt(((cur - tr) ** 2).mean()))
    rmse_pde = float(torch.sqrt(((pde_cur - tr) ** 2).mean()))
    rmse_per = float(torch.sqrt(((TX - tr) ** 2).mean()))

    if physics:
        g = torch.sigmoid(net.gate).detach().reshape(G, G)
        gm = float(g.mean())
        rows = g.mean(1)
        gr = float(rows.max() - rows.min())
    else:
        gm, gr = 0.0, 0.0

    return {"rmse": rmse, "rmse_pde_only": rmse_pde, "rmse_persistence": rmse_per,
            "gate_mean": gm, "gate_row_range": gr}
