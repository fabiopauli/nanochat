"""
NSCT v16.2 — Multi-Seed Validation
====================================

PROBLEM: v16 and v16.1 showed huge variance between runs.
  cap=0.15: 79.9% avg (v16) vs 72.8% avg (v16.1) — same architecture!
  This means initialization noise dominates the cap sweep signal.

SOLUTION: Run the two best caps with multiple seeds to get reliable estimates.

BEST CANDIDATES:
  cap=0.12 — hit hop 80.9% (extraordinary), comp 76.4%, avg 78.7%
  cap=0.15 — hit comp 84.5%, hop 75.2%, avg 79.9% (v16 run)

PROTOCOL:
  2 caps × 3 seeds × 2 tasks = 12 runs
  8,000 steps each (sweet spot from v16.1 curves)
  Seeds: 0, 1, 2 (torch + random)
  Report mean ± std across seeds

HISTORICAL REFERENCE:
  v9 best:        comp 81.2%, hop 60.3%, avg 70.7%
  v16-cap0.15:    comp 84.5%, hop 75.2%, avg 79.9% (single run)
  v16.1-cap0.12:  comp 76.4%, hop 80.9%, avg 78.7% (single run)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass


@dataclass
class Config:
    d_model: int     = 64
    n_heads: int     = 4
    n_layers: int    = 3
    max_seq_len: int = 20
    vocab_size: int  = 200
    n_entities: int  = 8
    dropout: float   = 0.1
    contrastive_alpha: float = 0.5
    contrastive_temp: float  = 0.1
    entity_temp: float = 0.5
    alive_threshold: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────────────────────────────────────

class CompositionalTask:
    def __init__(self, n_subjects=40, n_relations=8, n_objects=40,
                 n_classes=8, test_holdout=0.30, n_exemplars=3,
                 guarantee_same_class=False, seed=42):
        random.seed(seed)
        self.n_subjects  = n_subjects
        self.n_relations = n_relations
        self.n_objects   = n_objects
        self.n_classes   = n_classes
        self.n_exemplars = n_exemplars
        self.guarantee   = guarantee_same_class

        self.S = 0
        self.R = n_subjects
        self.O = n_subjects + n_relations
        self.QUERY = n_subjects + n_relations + n_objects
        self.total_vocab = self.QUERY + 1

        self.subj_class = [i % n_classes for i in range(n_subjects)]
        self.obj_class  = [i % n_classes for i in range(n_objects)]
        self.rel_table = [
            [random.randint(0, n_classes - 1) for _ in range(n_classes)]
            for _ in range(n_relations)
        ]
        self._class_obj_cache = {}
        for sc in range(n_classes):
            for r in range(n_relations):
                oc = self.rel_table[r][sc]
                objs_in_class = [o for o in range(n_objects) if self.obj_class[o] == oc]
                idx = (sc * 7 + r * 13) % len(objs_in_class)
                self._class_obj_cache[(sc, r)] = objs_in_class[idx]

        all_pairs = [(s, r) for s in range(n_subjects) for r in range(n_relations)]
        random.shuffle(all_pairs)
        split = int(len(all_pairs) * (1 - test_holdout))
        self.train_pairs = set(all_pairs[:split])
        self.test_pairs  = set(all_pairs[split:])
        self.exemplar_subjects = {}
        for r in range(n_relations):
            self.exemplar_subjects[r] = [s for s in range(n_subjects) if (s, r) in self.train_pairs]
        self._train_by_rel_class = {}
        for r in range(n_relations):
            for sc in range(n_classes):
                self._train_by_rel_class[(r, sc)] = [
                    s for s in self.exemplar_subjects[r] if self.subj_class[s] == sc
                ]

        self.exemplar_subj_positions = [i * 3 for i in range(n_exemplars)]
        self.exemplar_out_positions  = [i * 3 + 2 for i in range(n_exemplars)]
        self.query_subj_position     = n_exemplars * 3

    def object_for(self, s, r):
        return self._class_obj_cache[(self.subj_class[s], r)]

    def _make_sample(self, pair_set):
        for _ in range(500):
            s_q, r_q = random.choice(list(pair_set))
            o_q = self.object_for(s_q, r_q)
            sc_q = self.subj_class[s_q]
            exs_all = [s for s in self.exemplar_subjects[r_q] if s != s_q]
            if len(exs_all) < self.n_exemplars:
                continue
            if self.guarantee:
                same_class = [s for s in self._train_by_rel_class.get((r_q, sc_q), []) if s != s_q]
                if not same_class:
                    continue
                anchor = random.choice(same_class)
                others = [s for s in exs_all if s != anchor]
                if len(others) < self.n_exemplars - 1:
                    continue
                rest = random.sample(others, self.n_exemplars - 1)
                chosen = [anchor] + rest
                random.shuffle(chosen)
            else:
                chosen = random.sample(exs_all, self.n_exemplars)
            seq = []
            for ex in chosen:
                seq.extend([self.S + ex, self.R + r_q, self.O + self.object_for(ex, r_q)])
            seq.extend([self.S + s_q, self.R + r_q, self.QUERY])
            return seq, self.O + o_q
        raise RuntimeError("Could not sample")

    def make_batch(self, batch_size, split="train"):
        pair_set = self.train_pairs if split == "train" else self.test_pairs
        seqs, targets = [], []
        for _ in range(batch_size):
            seq, tgt = self._make_sample(pair_set)
            seqs.append(seq)
            targets.append(tgt)
        return torch.tensor(seqs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class MultiHopTask:
    def __init__(self, n_subjects=30, n_intermediates=10, n_objects=20,
                 n_relations=4, n_classes=5, test_holdout_frac=0.30,
                 guarantee_same_class=False, seed=99):
        random.seed(seed)
        self.n_subj = n_subjects
        self.n_mid  = n_intermediates
        self.n_obj  = n_objects
        self.n_rel  = n_relations
        self.n_cls  = n_classes
        self.guarantee = guarantee_same_class

        self.S_off = 0
        self.M_off = n_subjects
        self.O_off = n_subjects + n_intermediates
        self.R_off = n_subjects + n_intermediates + n_objects
        self.QUERY = self.R_off + n_relations
        self.total_vocab = self.QUERY + 1

        self.s_cls = [i % n_classes for i in range(n_subjects)]
        self.m_cls = [i % n_classes for i in range(n_intermediates)]
        self.o_cls = [i % n_classes for i in range(n_objects)]
        self.r1_table = [random.randint(0, n_classes - 1) for _ in range(n_classes)]
        self.r2_table = [random.randint(0, n_classes - 1) for _ in range(n_classes)]

        all_mids = list(range(n_intermediates))
        all_objs = list(range(n_objects))
        self._class_mid = {}
        self._class_obj = {}
        for sc in range(n_classes):
            mc = self.r1_table[sc]
            oc = self.r2_table[mc]
            mid_cands = [m for m in all_mids if self.m_cls[m] == mc] or all_mids
            obj_cands = [o for o in all_objs if self.o_cls[o] == oc] or all_objs
            self._class_mid[sc] = mid_cands[sc * 3 % len(mid_cands)]
            self._class_obj[sc] = obj_cands[sc * 7 % len(obj_cands)]

        self._mid = {s: self._class_mid[self.s_cls[s]] for s in range(n_subjects)}
        self._obj = {s: self._class_obj[self.s_cls[s]] for s in range(n_subjects)}

        subjs = list(range(n_subjects))
        random.shuffle(subjs)
        split = int(n_subjects * (1 - test_holdout_frac))
        self.train_subjs = set(subjs[:split])
        self.test_subjs  = set(subjs[split:])
        self._train_by_class = {}
        for sc in range(n_classes):
            self._train_by_class[sc] = [s for s in self.train_subjs if self.s_cls[s] == sc]

        self.exemplar_subj_positions = [0, 6]
        self.exemplar_out_positions  = [2, 8]
        self.query_subj_position     = 12

    def make_batch(self, batch_size, split="train"):
        pool = self.train_subjs if split == "train" else self.test_subjs
        seqs, targets = [], []
        for _ in range(batch_size):
            s_q = random.choice(list(pool))
            m_q = self._mid[s_q]
            sc_q = self.s_cls[s_q]
            ex_pool = [s for s in self.train_subjs if s != s_q]
            if self.guarantee:
                same_class = [s for s in self._train_by_class.get(sc_q, []) if s != s_q]
                if not same_class:
                    ex1, ex2 = random.sample(ex_pool, 2)
                else:
                    ex1 = random.choice(same_class)
                    others = [s for s in ex_pool if s != ex1]
                    ex2 = random.choice(others)
                    if random.random() < 0.5:
                        ex1, ex2 = ex2, ex1
            else:
                ex1, ex2 = random.sample(ex_pool, 2)
            seq = [
                self.S_off + ex1, self.R_off + 0, self.M_off + self._mid[ex1],
                self.M_off + self._mid[ex1], self.R_off + 1, self.O_off + self._obj[ex1],
                self.S_off + ex2, self.R_off + 0, self.M_off + self._mid[ex2],
                self.M_off + self._mid[ex2], self.R_off + 1, self.O_off + self._obj[ex2],
                self.S_off + s_q, self.R_off + 0, self.QUERY,
            ]
            seqs.append(seq)
            targets.append(self.M_off + m_q)
        return torch.tensor(seqs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, entity_bias=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        def sh(t): return t.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if entity_bias is not None:
            w = w + entity_bias.unsqueeze(1)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        w = w.masked_fill(causal[None, None], -1e9)
        out = torch.matmul(F.softmax(w, -1), v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attn  = CausalAttention(cfg)
        self.ffn   = FeedForward(cfg)

    def forward(self, h, entity_bias=None):
        h = h + self.attn(self.norm1(h), entity_bias=entity_bias)
        h = h + self.ffn(self.norm2(h))
        return h


class EntityProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.n_entities),
        )
        self.temp = cfg.entity_temp

    def forward(self, h, hard=False):
        logits = self.net(h)
        if hard:
            idx = logits.argmax(-1, keepdim=True)
            y = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
            return y - logits.detach() + logits
        return F.softmax(logits / self.temp, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# v16 MODEL
# ─────────────────────────────────────────────────────────────────────────────

class FlexTransformer(nn.Module):
    def __init__(self, cfg, scale0_cap=0.15):
        super().__init__()
        self.cfg = cfg
        self.use_binding = True
        self.use_contrastive = True
        self.inject_mode = 'progressive'
        self.scale0_cap = scale0_cap
        self.use_slot_attn = False
        self.use_inject = True

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.entity_norm = nn.LayerNorm(cfg.d_model)
        self.entity_proj = EntityProjection(cfg)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

        self.inject_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)),  # scale₀
            nn.Parameter(torch.tensor(0.0)),  # scale₁
        ])

        self.ex_out_pos = None
        self.q_subj_pos = None
        self.ex_subj_pos = None

    def set_task_positions(self, task):
        self.ex_out_pos = task.exemplar_out_positions
        self.q_subj_pos = task.query_subj_position
        self.ex_subj_pos = task.exemplar_subj_positions

    def forward(self, x, targets=None, hard=False):
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T]

        h = self.blocks[0](h)

        ep0 = self.entity_proj(self.entity_norm(h), hard=hard)
        bias0 = torch.bmm(ep0, ep0.transpose(-1, -2))

        s0 = self.inject_scales[0].clamp(min=0.0, max=self.scale0_cap)
        h = self.blocks[1](h, entity_bias=bias0 * s0)

        ep1 = self.entity_proj(self.entity_norm(h), hard=hard)
        bias1 = torch.bmm(ep1, ep1.transpose(-1, -2))

        s1 = self.inject_scales[1]
        h = self.blocks[2](h, entity_bias=bias1 * s1)

        ep = ep1

        assign_t = ep.transpose(-1, -2)
        weights = assign_t / (assign_t.sum(-1, keepdim=True) + 1e-8)
        slots = torch.bmm(weights, h)
        readout = torch.bmm(ep, slots)
        g = torch.sigmoid(self.gate_logit)
        h = (1.0 - g) * h + g * readout

        logits = self.head(self.final_norm(h[:, -1]))

        slot_mass = ep.sum(dim=1)
        slot_frac = slot_mass / (slot_mass.sum(dim=-1, keepdim=True) + 1e-8)
        n_alive = (slot_frac > self.cfg.alive_threshold).float().sum(dim=-1).mean()

        if targets is not None:
            task_loss = F.cross_entropy(logits, targets)
            cl = self._contrastive_loss(x, ep, targets)
            total = task_loss + self.cfg.contrastive_alpha * cl
            return logits, total, task_loss, cl, g, n_alive, s0.item(), s1.item()

        return logits

    def _contrastive_loss(self, x, ep, targets):
        B = x.shape[0]
        device = x.device
        if self.ex_out_pos is None:
            return torch.tensor(0.0, device=device)
        ep_query = ep[:, self.q_subj_pos, :]
        ep_exs = ep[:, self.ex_subj_pos, :]
        ex_outs = x[:, self.ex_out_pos]
        same = (ex_outs == targets.unsqueeze(1)).float()
        has_match = same.sum(-1) > 0
        if not has_match.any():
            return torch.tensor(0.0, device=device)
        sim = torch.bmm(ep_exs, ep_query.unsqueeze(-1)).squeeze(-1)
        target_dist = same / (same.sum(-1, keepdim=True) + 1e-8)
        log_probs = F.log_softmax(sim / self.cfg.contrastive_temp, dim=-1)
        loss = -(target_dist * log_probs).sum(-1)
        return loss[has_match].mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def evaluate_split(model, task, split, n_batches=50, batch_size=64, device='cpu'):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = task.make_batch(batch_size, split=split)
            x, y = x.to(device), y.to(device)
            logits = model(x, hard=True)
            c += (logits.argmax(-1) == y).sum().item()
            t += y.size(0)
    model.train()
    return c / t


def train_model(model, task, n_steps=8000, batch_size=64,
                lr=3e-4, eval_every=1000, label="", device='cpu'):
    model = model.to(device)
    model.set_task_positions(task)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.1)

    print(f"  Training {label}...", end=" ", flush=True)

    best_test = 0.0
    for step in range(1, n_steps + 1):
        x, y = task.make_batch(batch_size, split="train")
        x, y = x.to(device), y.to(device)

        logits, total_loss, task_loss, cl, gate_val, n_alive, s0, s1 = model(x, targets=y)

        opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0:
            te = evaluate_split(model, task, "test", device=device)
            best_test = max(best_test, te)

    te_f = evaluate_split(model, task, "test", n_batches=100, device=device)
    best_test = max(best_test, te_f)
    
    # Get final scale values
    s0_final = model.inject_scales[0].clamp(min=0.0, max=model.scale0_cap).item()
    s1_final = model.inject_scales[1].item()
    
    print(f"best={best_test:.2%}  final={te_f:.2%}  s0={s0_final:.4f}  s1={s1_final:.4f}")
    return te_f, best_test, s0_final, s1_final


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 100)
    print("  NSCT v16.2 — Multi-Seed Validation")
    print("=" * 100)
    print()
    print("  Protocol: 2 caps × 3 seeds × 2 tasks = 12 runs @ 8k steps")
    print("  Caps: 0.12, 0.15")
    print("  Seeds: 0, 1, 2")
    print()
    print("  Goal: Measure true expected performance with confidence intervals")
    print("  Reference: v9 avg 70.7%, v16 best single-run avg 79.9%")
    print()

    caps = [0.12, 0.15]
    seeds = [0, 1, 2]
    n_steps = 8000

    # ── Compositional Task ──
    # Task uses seed=42 internally for data splits (always the same)
    
    print("=" * 100)
    print("  COMPOSITIONAL TASK")
    print("=" * 100)
    
    comp_results = {}
    for cap in caps:
        comp_results[cap] = []
        for seed in seeds:
            torch.manual_seed(seed)
            random.seed(seed)
            
            task = CompositionalTask(
                n_subjects=40, n_relations=8, n_objects=40, n_classes=8,
                test_holdout=0.30, n_exemplars=3, guarantee_same_class=True, seed=42)
            cfg = Config()
            cfg.vocab_size = task.total_vocab
            
            # Reset random seed for training sampling after task construction
            random.seed(seed * 1000 + 1)
            
            model = FlexTransformer(cfg, scale0_cap=cap)
            label = f"cap={cap:.2f} seed={seed}"
            te_f, best, s0, s1 = train_model(
                model, task, n_steps=n_steps, label=label, device=device)
            comp_results[cap].append({"final": te_f, "best": best, "s0": s0, "s1": s1, "seed": seed})

    print(f"\n{'─'*80}")
    print(f"  Compositional Results")
    print(f"{'─'*80}")
    for cap in caps:
        bests = [r["best"] for r in comp_results[cap]]
        mean_b = sum(bests) / len(bests)
        std_b = (sum((b - mean_b)**2 for b in bests) / len(bests)) ** 0.5
        finals = [r["final"] for r in comp_results[cap]]
        mean_f = sum(finals) / len(finals)
        s1s = [r["s1"] for r in comp_results[cap]]
        mean_s1 = sum(s1s) / len(s1s)
        print(f"\n  cap={cap:.2f}:")
        for r in comp_results[cap]:
            print(f"    seed={r['seed']}: best={r['best']:.2%}  final={r['final']:.2%}  "
                  f"s0={r['s0']:.4f}  s1={r['s1']:.4f}")
        print(f"    ────────────────────────────────────────")
        print(f"    MEAN best: {mean_b:.2%} ± {std_b:.2%}   final: {mean_f:.2%}   s1: {mean_s1:.4f}")

    # ── Multi-Hop Task ──
    
    print(f"\n{'='*100}")
    print("  MULTI-HOP TASK")
    print("=" * 100)
    
    hop_results = {}
    for cap in caps:
        hop_results[cap] = []
        for seed in seeds:
            torch.manual_seed(seed)
            random.seed(seed)
            
            task = MultiHopTask(
                n_subjects=30, n_intermediates=10, n_objects=20, n_relations=4,
                n_classes=5, test_holdout_frac=0.30, guarantee_same_class=True, seed=99)
            cfg = Config()
            cfg.vocab_size = task.total_vocab
            
            random.seed(seed * 1000 + 2)
            
            model = FlexTransformer(cfg, scale0_cap=cap)
            label = f"cap={cap:.2f} seed={seed}"
            te_f, best, s0, s1 = train_model(
                model, task, n_steps=n_steps, label=label, device=device)
            hop_results[cap].append({"final": te_f, "best": best, "s0": s0, "s1": s1, "seed": seed})

    print(f"\n{'─'*80}")
    print(f"  Multi-Hop Results")
    print(f"{'─'*80}")
    for cap in caps:
        bests = [r["best"] for r in hop_results[cap]]
        mean_b = sum(bests) / len(bests)
        std_b = (sum((b - mean_b)**2 for b in bests) / len(bests)) ** 0.5
        finals = [r["final"] for r in hop_results[cap]]
        mean_f = sum(finals) / len(finals)
        s1s = [r["s1"] for r in hop_results[cap]]
        mean_s1 = sum(s1s) / len(s1s)
        print(f"\n  cap={cap:.2f}:")
        for r in hop_results[cap]:
            print(f"    seed={r['seed']}: best={r['best']:.2%}  final={r['final']:.2%}  "
                  f"s0={r['s0']:.4f}  s1={r['s1']:.4f}")
        print(f"    ────────────────────────────────────────")
        print(f"    MEAN best: {mean_b:.2%} ± {std_b:.2%}   final: {mean_f:.2%}   s1: {mean_s1:.4f}")

    # ── Grand Summary ──
    
    print(f"\n{'='*100}")
    print("  v16.2 GRAND SUMMARY — Multi-Seed Validation")
    print(f"{'='*100}")
    
    print(f"\n  {'Cap':>5}  {'Comp (mean±std)':>20}  {'Hop (mean±std)':>20}  "
          f"{'Avg':>8}  {'vs v9':>8}")
    print(f"  {'─'*5}  {'─'*20}  {'─'*20}  {'─'*8}  {'─'*8}")
    
    best_overall_avg = 0
    best_overall_cap = 0
    
    for cap in caps:
        comp_bests = [r["best"] for r in comp_results[cap]]
        hop_bests = [r["best"] for r in hop_results[cap]]
        
        comp_mean = sum(comp_bests) / len(comp_bests)
        comp_std = (sum((b - comp_mean)**2 for b in comp_bests) / len(comp_bests)) ** 0.5
        hop_mean = sum(hop_bests) / len(hop_bests)
        hop_std = (sum((b - hop_mean)**2 for b in hop_bests) / len(hop_bests)) ** 0.5
        
        avg = (comp_mean + hop_mean) / 2
        v9_delta = avg - 0.7073
        
        comp_str = f"{comp_mean:.1%} ± {comp_std:.1%}"
        hop_str = f"{hop_mean:.1%} ± {hop_std:.1%}"
        
        print(f"  {cap:>5.2f}  {comp_str:>20}  {hop_str:>20}  {avg:>8.2%}  {v9_delta:>+7.2%}")
        
        if avg > best_overall_avg:
            best_overall_avg = avg
            best_overall_cap = cap

    # Individual seed averages
    print(f"\n  Per-seed averages:")
    print(f"  {'Cap':>5}  {'Seed':>5}  {'Comp':>7}  {'Hop':>7}  {'Avg':>7}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}")
    for cap in caps:
        for i, seed in enumerate(seeds):
            c = comp_results[cap][i]["best"]
            h = hop_results[cap][i]["best"]
            a = (c + h) / 2
            print(f"  {cap:>5.2f}  {seed:>5}  {c:>7.1%}  {h:>7.1%}  {a:>7.1%}")
        print()

    print(f"\n  🏆 Best cap (by mean avg): {best_overall_cap:.2f} → {best_overall_avg:.2%}")
    print(f"  📊 v9 reference: 70.73%")
    print(f"  📊 v16 best single-run: 79.88%")
    print(f"  Δ vs v9: {best_overall_avg - 0.7073:+.2%}")
    
    # Consistency check
    print(f"\n  Consistency analysis:")
    for cap in caps:
        comp_bests = [r["best"] for r in comp_results[cap]]
        hop_bests = [r["best"] for r in hop_results[cap]]
        comp_range = max(comp_bests) - min(comp_bests)
        hop_range = max(hop_bests) - min(hop_bests)
        print(f"    cap={cap:.2f}: comp range={comp_range:.1%}, hop range={hop_range:.1%}")

    print("=" * 100)


if __name__ == "__main__":
    main()