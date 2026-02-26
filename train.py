import os
os.environ['MUJOCO_GL'] = 'egl'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dm_control import suite
from torch.utils.data import TensorDataset, DataLoader
import pickle

# --- CONFIG ---
SEEDS = [0, 1, 2, 3, 4]
ENVS = [
    ('pendulum', 'swingup'),
    ('cartpole', 'balance'),
    ('acrobot', 'swingup'),
    ('walker', 'walk'),
    ('cheetah', 'run'),
    ('hopper', 'hop')
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
DATA_DIR = "bench_final"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Running Benchmark on {DEVICE}")

# ==========================================
# 1. UTILS
# ==========================================
def make_invariant(q, env_name):
    q_inv = q.clone()
    if any(x in env_name for x in ['cartpole', 'walker', 'cheetah', 'hopper']):
        if q.dim() == 1: q_inv[0] = 0.0
        else: q_inv[:, 0] = 0.0
    return q_inv

class Normalizer:
    def __init__(self, dim):
        self.mean = torch.zeros(dim).to(DEVICE)
        self.std = torch.ones(dim).to(DEVICE)
    def fit(self, x):
        self.mean = x.mean(0); self.std = x.std(0) + 1e-5
    def normalize(self, x): return (x - self.mean) / self.std
    def unnormalize(self, x): return (x * self.std) + self.mean

# ==========================================
# 2. MODEL ZOO (With Safety Clamps)
# ==========================================

# Standard MLP with Output Constraint (Prevents 1e14 explosion)
class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, d_out)
        )
    def forward(self, x):
        # Soft clamp output to reasonable physics range (-100, 100 in norm space)
        return 10.0 * torch.tanh(self.net(x) / 10.0)

class Ensemble(nn.Module):
    def __init__(self, d_in, d_out, n_models=5):
        super().__init__()
        self.models = nn.ModuleList([MLP(d_in, d_out) for _ in range(n_models)])
    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models])
        return preds.mean(0), preds.var(0)

class GRUModel(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.gru = nn.GRU(d_in, 256, batch_first=True)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, d_out))
    def forward(self, x):
        out, _ = self.gru(x)
        # Soft clamp
        return 10.0 * torch.tanh(self.head(out[:, -1, :]) / 10.0)

class LNN(nn.Module):
    def __init__(self, n_q, n_v):
        super().__init__()
        self.n_q = n_q; self.n_v = n_v
        self.l_net = nn.Sequential(nn.Linear(n_q, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, (n_v*(n_v+1))//2))
        self.v_net = nn.Sequential(nn.Linear(n_q, 256), nn.Tanh(), nn.Linear(256, 1))
        # Projection if dims mismatch
        self.proj = nn.Linear(n_q, n_v, bias=False) if n_q != n_v else nn.Identity()

    def get_M(self, q):
        bs = q.shape[0]; l = self.l_net(q)
        L = torch.zeros(bs, self.n_v, self.n_v, device=q.device)
        idx = torch.tril_indices(self.n_v, self.n_v)
        L[:, idx[0], idx[1]] = l; diag = torch.arange(self.n_v)
        L[:, diag, diag] = F.softplus(L[:, diag, diag]) + 1.0
        return torch.bmm(L, L.transpose(1, 2))

    def forward(self, q, qd, tau):
        with torch.enable_grad():
            q.requires_grad_(True); qd.requires_grad_(True)
            M = self.get_M(q); T = 0.5 * (qd.unsqueeze(1) @ M @ qd.unsqueeze(2)).squeeze()
            V = self.v_net(q).squeeze(); L_sys = T - V
            
            p = torch.autograd.grad(L_sys.sum(), qd, create_graph=True)[0]
            dL_dq = torch.autograd.grad(L_sys.sum(), q, create_graph=True)[0]
            
            # Dimension matching
            dp_dt = self.proj(dL_dq) if self.n_q != self.n_v else dL_dq
            p_grad = torch.autograd.grad((p * qd.detach()).sum(), q, create_graph=True)[0]
            term1 = self.proj(p_grad) if self.n_q != self.n_v else p_grad
            
            acc = torch.linalg.solve(M, (tau - term1 + dp_dt).unsqueeze(2)).squeeze(2)
            return torch.clamp(acc, -100.0, 100.0)

class PhysicsConditionedFlow(nn.Module):
    def __init__(self, d_state, d_cond, d_hint):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+d_state+d_cond+d_hint, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, d_state)
        )
        self.net[-1].weight.data.zero_() 
    def forward(self, t, x, cond, hint):
        return self.net(torch.cat([t, x, cond, torch.tanh(hint/10.0)], dim=1))

# ==========================================
# 3. TRAINING
# ==========================================
def train_seed(env_name, task_name, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    
    try: env = suite.load(env_name, task_name)
    except: return None
    
    n_q, n_v, n_u = env.physics.data.qpos.shape[0], env.physics.data.qvel.shape[0], env.action_spec().shape[0]
    
    # Collect
    steps = 30000
    data, ts = [], env.reset()
    hist = np.zeros((3, n_q+n_v+n_u))
    
    for _ in range(steps):
        q, qd = env.physics.data.qpos.copy(), env.physics.data.qvel.copy()
        u = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)
        q_inv = make_invariant(torch.tensor(q, dtype=torch.float32), env_name).numpy()
        hist[:-1] = hist[1:]; hist[-1] = np.concatenate([q_inv, qd, u])
        env.step(u)
        qdd = (env.physics.data.qvel.copy() - qd) / env.control_timestep()
        data.append((hist.copy(), qdd))
        if ts.last(): env.reset(); hist[:] = 0
        
    X_np = np.array([x[0] for x in data]).reshape(len(data), -1)
    Y_np = np.array([x[1] for x in data])
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=DEVICE)
    
    norm_in = Normalizer(X.shape[1]); norm_in.fit(X)
    norm_out = Normalizer(n_v); norm_out.fit(Y)
    loader = DataLoader(TensorDataset(norm_in.normalize(X), norm_out.normalize(Y)), batch_size=4096, shuffle=True)
    
    # Init Models
    mlp = MLP(X.shape[1], n_v).to(DEVICE)
    ens = Ensemble(X.shape[1], n_v).to(DEVICE)
    gru = GRUModel(n_q+n_v+n_u, n_v).to(DEVICE)
    lnn_pure = LNN(n_q, n_v).to(DEVICE)
    lnn_hyb = LNN(n_q, n_v).to(DEVICE)
    flow = PhysicsConditionedFlow(n_v, X.shape[1], n_v).to(DEVICE)
    
    opts = [
        optim.Adam(mlp.parameters(), lr=1e-3),
        optim.Adam(ens.parameters(), lr=1e-3),
        optim.Adam(gru.parameters(), lr=1e-3),
        optim.Adam(lnn_pure.parameters(), lr=5e-4, weight_decay=1e-4),
        optim.Adam(lnn_hyb.parameters(), lr=5e-4, weight_decay=1e-4),
        optim.Adam(flow.parameters(), lr=1e-3)
    ]
    
    print(f"   🌱 Seed {seed}: Training...")
    for ep in range(150):
        for bx, by in loader:
            # MLP/Ens/GRU
            loss_m = F.mse_loss(mlp(bx), by)
            opts[0].zero_grad(); loss_m.backward(); opts[0].step()
            
            preds = torch.stack([m(bx) for m in ens.models])
            l_e = F.mse_loss(preds, by.unsqueeze(0).repeat(5,1,1))
            opts[1].zero_grad(); l_e.backward(); opts[1].step()
            
            bx_seq = bx.reshape(bx.shape[0], 3, -1)
            l_g = F.mse_loss(gru(bx_seq), by)
            opts[2].zero_grad(); l_g.backward(); opts[2].step()
            
            # Physics
            curr = bx_seq[:, -1, :]
            q, qd, u = curr[:,:n_q], curr[:,n_q:n_q+n_v], curr[:,-n_u:]
            tau = torch.zeros_like(q[:, :n_v]); tau[:, :n_u] = u if n_u < n_v else u
            
            # Pure LNN
            l_lp = F.mse_loss(lnn_pure(q, qd, tau), by)
            opts[3].zero_grad(); l_lp.backward(); opts[3].step()
            
            # Hybrid
            hint = lnn_hyb(q, qd, tau)
            if ep < 30: 
                l_lh = F.mse_loss(hint, by)
                opts[4].zero_grad(); l_lh.backward(); opts[4].step()
            
            t = torch.rand(bx.shape[0], 1, device=DEVICE)
            x0 = torch.randn_like(by)
            xt = t*by + (1-t)*x0
            vt = flow(t, xt, bx, hint.detach())
            l_f = F.mse_loss(vt, by - x0)
            opts[5].zero_grad(); l_f.backward(); opts[5].step()

    sd = f"{DATA_DIR}/{env_name}_{seed}"
    os.makedirs(sd, exist_ok=True)
    torch.save(mlp.state_dict(), f"{sd}/mlp.pt")
    torch.save(ens.state_dict(), f"{sd}/ens.pt")
    torch.save(gru.state_dict(), f"{sd}/gru.pt")
    torch.save(lnn_pure.state_dict(), f"{sd}/lnn_pure.pt")
    torch.save(lnn_hyb.state_dict(), f"{sd}/lnn_hyb.pt")
    torch.save(flow.state_dict(), f"{sd}/flow.pt")
    with open(f"{sd}/norm.pkl", 'wb') as f: pickle.dump([norm_in.mean, norm_in.std, norm_out.mean, norm_out.std], f)
    return (n_q, n_v, n_u)

# ==========================================
# 4. EVALUATION
# ==========================================
def eval_seed(env_name, task_name, seed, dims):
    np.random.seed(seed); torch.manual_seed(seed)
    sd = f"{DATA_DIR}/{env_name}_{seed}"
    n_q, n_v, n_u = dims
    
    # Load
    mlp = MLP(3*(n_q+n_v+n_u), n_v).to(DEVICE)
    ens = Ensemble(3*(n_q+n_v+n_u), n_v).to(DEVICE)
    gru = GRUModel(n_q+n_v+n_u, n_v).to(DEVICE)
    lnn_pure = LNN(n_q, n_v).to(DEVICE)
    lnn_hyb = LNN(n_q, n_v).to(DEVICE)
    flow = PhysicsConditionedFlow(n_v, 3*(n_q+n_v+n_u), n_v).to(DEVICE)
    
    mlp.load_state_dict(torch.load(f"{sd}/mlp.pt"))
    ens.load_state_dict(torch.load(f"{sd}/ens.pt"))
    gru.load_state_dict(torch.load(f"{sd}/gru.pt"))
    lnn_pure.load_state_dict(torch.load(f"{sd}/lnn_pure.pt"))
    lnn_hyb.load_state_dict(torch.load(f"{sd}/lnn_hyb.pt"))
    flow.load_state_dict(torch.load(f"{sd}/flow.pt"))
    
    with open(f"{sd}/norm.pkl", 'rb') as f: m_in, s_in, m_out, s_out = pickle.load(f)
    
    # --- A. ROLLOUT ---
    env = suite.load(env_name, task_name); env.reset()
    hist = np.zeros((3, n_q+n_v+n_u))
    for _ in range(3):
        u = np.zeros(n_u); env.step(u)
        qi = make_invariant(torch.tensor(env.physics.data.qpos, dtype=torch.float32), env_name).numpy()
        hist[:-1] = hist[1:]; hist[-1] = np.concatenate([qi, env.physics.data.qvel, u])
    
    # Init states for all models
    states = {k: {'q': env.physics.data.qpos.copy(), 'qd': env.physics.data.qvel.copy(), 'hist': hist.copy(), 'traj': []} 
              for k in ['gt', 'mlp', 'ens', 'gru', 'lnn', 'hyb']}
    
    actions = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, size=(150, n_u))
    
    for i in range(150):
        u = actions[i]; dt = env.control_timestep()
        env.step(u); states['gt']['traj'].append(env.physics.data.qpos[0])
        u_t = torch.tensor(u, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            for m in ['mlp', 'ens', 'gru', 'lnn', 'hyb']:
                st = states[m]
                h_flat = torch.tensor(st['hist'].flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                in_n = (h_flat - m_in) / s_in
                
                if m == 'mlp': acc = mlp(in_n)
                elif m == 'ens': acc, _ = ens(in_n)
                elif m == 'gru': acc = gru(in_n.reshape(1, 3, -1))
                elif m == 'lnn':
                    curr = in_n.reshape(1, 3, -1)[:, -1, :]
                    q, qd = curr[:,:n_q], curr[:,n_q:n_q+n_v]
                    tau = torch.zeros_like(q[:, :n_v]); tau[:,:n_u] = u_t
                    acc = lnn_pure(q, qd, tau) 
                elif m == 'hyb':
                    curr = in_n.reshape(1, 3, -1)[:, -1, :]
                    q, qd = curr[:,:n_q], curr[:,n_q:n_q+n_v]
                    tau = torch.zeros_like(q[:, :n_v]); tau[:,:n_u] = u_t
                    hint = lnn_hyb(q, qd, tau)
                    xc = torch.randn_like(hint)
                    for k in range(2):
                        v = flow(torch.ones(1,1,device=DEVICE)*(k/2.0), xc, in_n, hint)
                        xc += v * 0.5
                    acc = xc
                
                real_acc = (acc * s_out + m_out).cpu().numpy()[0]
                real_acc = np.clip(real_acc, -500, 500) # Hard clamp for rollout stability
                st['qd'] += real_acc * dt; st['q'] += st['qd'] * dt
                st['traj'].append(st['q'][0])
                qi = make_invariant(torch.tensor(st['q'], dtype=torch.float32), env_name).numpy()
                st['hist'][:-1] = st['hist'][1:]; st['hist'][-1] = np.concatenate([qi, st['qd'], u])

    # --- B. BIFURCATION ---
    bif_vars = {'mlp': 0., 'ens': 0., 'hyb': 0.}
    # We calculate bifurcation for ALL environments now, not just pendulums
    # Just force a zero-velocity state
    ts = env.reset()
    with env.physics.reset_context():
        env.physics.data.qvel[:] = 0.0
    
    qi = make_invariant(torch.tensor(env.physics.data.qpos, dtype=torch.float32), env_name).numpy()
    state = np.concatenate([qi, env.physics.data.qvel, np.zeros(n_u)])
    h_flat = torch.tensor(np.tile(state, (3, 1)).flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    in_n = (h_flat - m_in) / s_in
    
    with torch.no_grad():
        _, var = ens(in_n); bif_vars['ens'] = var.mean().item()
        curr = in_n.reshape(1, 3, -1)[:, -1, :]
        hint = lnn_hyb(curr[:,:n_q], curr[:,n_q:n_q+n_v], torch.zeros(1, n_v, device=DEVICE))
        preds = []
        for _ in range(50):
            xc = torch.randn_like(hint)
            for k in range(5):
                v = flow(torch.ones(1,1,device=DEVICE)*(k/5.0), xc, in_n, hint)
                xc += v * 0.2
            preds.append(xc[0, -1].item())
        bif_vars['hyb'] = np.std(preds)

    return {'traj': {k: v['traj'] for k, v in states.items()}, 'bif': bif_vars}

if __name__ == "__main__":
    final_res = {}
    for e, t in ENVS:
        print(f"\n{e}...")
        final_res[e] = []
        for s in SEEDS:
            dims = train_seed(e, t, s)
            if dims: final_res[e].append(eval_seed(e, t, s, dims))
    with open("results.pkl", "wb") as f: pickle.dump(final_res, f)
    print("Done.")