"""Deep Q-Network written in plain NumPy (no deep-learning library).

Pieces, in the order the course teaches them:
  * a small neural network Q(s, a) with hand-written backpropagation
  * the Adam optimiser
  * an experience-replay buffer
  * a slowly-updated target network (soft updates)
  * epsilon-greedy exploration

One network is shared by all small cells. Each cell feeds in its own view of
the world and gets back two numbers: the value of sleeping and of staying on.
"""
import numpy as np

from .config import Config
from .env import HetNetEnv, run_policy
from .traffic import generate


# --------------------------------------------------------------------------
# neural network
# --------------------------------------------------------------------------
class MLP:
    """Fully connected network with ReLU hidden layers and a linear output."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0.0, np.sqrt(2.0 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, x, keep=False):
        acts = [x]
        a = x
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if l < last:
                a = np.maximum(a, 0.0)
            acts.append(a)
        if keep:
            self._acts = acts
        return a

    def backward(self, d_out):
        """Gradients of the loss w.r.t. every weight, given d(loss)/d(output)."""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        delta = d_out
        for l in reversed(range(len(self.W))):
            gW[l] = self._acts[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (self._acts[l] > 0)
        return gW, gb

    def params(self):
        return self.W + self.b

    def copy_from(self, other, tau=1.0):
        """Soft update: self = tau * other + (1 - tau) * self."""
        for p, q in zip(self.params(), other.params()):
            p *= (1.0 - tau)
            p += tau * q


class Adam:
    def __init__(self, params, lr=5e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.k = 0

    def step(self, grads):
        self.k += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.k)
            v_hat = v / (1 - self.b2 ** self.k)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity, n_features):
        self.s = np.zeros((capacity, n_features))
        self.s2 = np.zeros((capacity, n_features))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def add(self, s, a, r, s2):
        n = len(a)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx] = s, a, r, s2
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, rng, batch):
        idx = rng.integers(0, self.size, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx]


# --------------------------------------------------------------------------
# what each cell tells the network
# --------------------------------------------------------------------------
N_BASE_FEATURES = 11


def featurize(obs, cfg: Config, onehot=None):
    """Build the (N, n_features) input, one row per small cell.

    Only past measurements are used. `onehot` (N, K) is the traffic type of
    each cell found by K-means; pass None to leave it out (ablation).
    """
    d = obs.lags[:, 0]
    n = len(d)
    cm = cfg.macro_capacity
    hour_angle = 2.0 * np.pi * obs.slot / cfg.steps_per_day
    # "If I and every cell quieter than me slept, how full would the macro be?"
    order = np.argsort(d)
    packed = np.empty(n)
    packed[order] = np.cumsum(np.minimum(d[order], cfg.small_capacity))
    overflow = float(np.maximum(d - cfg.small_capacity, 0.0).sum())
    pack = (obs.macro_prev + overflow + packed) / cm
    cols = [
        np.minimum(obs.lags, 2.0),                          # 3: own traffic, last 45 min
        np.full((n, 1), np.sin(hour_angle)),                # time of day
        np.full((n, 1), np.cos(hour_angle)),
        np.full((n, 1), float(obs.weekend)),
        obs.status[:, None].astype(float),                  # am I on right now?
        obs.locked[:, None].astype(float),                  # am I still in dwell time?
        np.full((n, 1), obs.macro_prev / cm),               # macro's own load
        np.full((n, 1), min(obs.util_prev, 1.5)),           # macro's total load
        np.minimum(pack, 2.0)[:, None],
    ]
    if onehot is not None:
        cols.append(onehot)
    return np.concatenate(cols, axis=1)


class DQNPolicy:
    """Greedy controller that uses a trained Q-network."""

    name = "DQN"

    def __init__(self, net, cfg: Config, onehot=None):
        self.net, self.cfg, self.onehot = net, cfg, onehot

    def act(self, obs):
        q = self.net.forward(featurize(obs, self.cfg, self.onehot))
        return np.argmax(q, axis=1).astype(np.int8)        # action 1 = on, 0 = asleep


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_dqn(cfg: Config, seed, onehot=None, epochs=60, days_per_epoch=14,
              hidden=(64, 64), gamma=0.95, lr=5e-4, batch=128, tau=0.01,
              eps_end=0.02, eps_decay_frac=0.5, buffer_size=200_000, grad_clip=10.0,
              val=None, eval_every=4, log=None):
    """Train one agent. Every epoch uses freshly simulated days.

    Returns (policy, history) where history holds validation scores over time.
    """
    rng = np.random.default_rng(seed)
    n_feat = N_BASE_FEATURES + (0 if onehot is None else onehot.shape[1])
    net = MLP((n_feat, *hidden, 2), rng)
    target = MLP((n_feat, *hidden, 2), rng)
    target.copy_from(net, tau=1.0)
    opt = Adam(net.params(), lr=lr)
    buf = ReplayBuffer(buffer_size, n_feat)
    policy = DQNPolicy(target, cfg, onehot)   # the target net is a running average of `net`: steadier

    steps_total = epochs * (days_per_epoch * cfg.steps_per_day - cfg.n_lags)
    step = 0
    history = []
    rows = np.arange(batch)
    for epoch in range(epochs):
        data = generate(cfg, days_per_epoch, seed=seed * 100_003 + epoch, start_weekday=epoch % 7)
        env = HetNetEnv(data, cfg)
        obs = env.reset()
        s = featurize(obs, cfg, onehot)
        while not env.done:
            eps = max(eps_end, 1.0 - (1.0 - eps_end) * step / (eps_decay_frac * steps_total))
            opt.lr = lr * (1.0 - 0.9 * step / steps_total)             # learning rate decays to 10%
            greedy = np.argmax(net.forward(s), axis=1)
            explore = rng.random(cfg.n_cells) < eps
            request = np.where(explore, rng.integers(0, 2, cfg.n_cells), greedy)
            reward, info = env.step(request)
            done = env.done
            if not done:
                obs = env.observe()
                s2 = featurize(obs, cfg, onehot)
                buf.add(s, info["status"], reward, s2)      # store the state that was really applied
                s = s2
            step += 1

            if buf.size >= 5_000:
                bs, ba, br, bs2 = buf.sample(rng, batch)
                y = br + gamma * target.forward(bs2).max(axis=1)      # Bellman target
                q = net.forward(bs, keep=True)
                td = q[rows, ba] - y
                d_out = np.zeros_like(q)
                d_out[rows, ba] = td / batch                          # gradient of 0.5 * mean(td^2)
                gW, gb = net.backward(d_out)
                grads = gW + gb
                norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
                if norm > grad_clip:                                  # keep rare huge penalties from blowing up a step
                    grads = [g * (grad_clip / norm) for g in grads]
                opt.step(grads)
                target.copy_from(net, tau=tau)                        # soft target update

        if val is not None and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            sc = run_policy(policy, val, cfg)
            history.append({"epoch": epoch + 1, **sc})
            if log:
                log(f"seed {seed} epoch {epoch + 1:3d}  energy {sc['energy_kwh_per_day']:.2f} kWh/day"
                    f"  unserved {sc['unserved_pct']:.4f}%")
    return policy, history
