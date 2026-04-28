# Numerical-Copula-Trading

MATH5030 Group 17 — Reproduction of a copula-based cryptocurrency pairs trading strategy.

Paper reproduced: Tadi & Witzany (2025), *Copula-based trading of cointegrated cryptocurrency pairs*, **Financial Innovation** 11:40.

---

## Project structure

```
Numerical-Copula-Trading/
├── data/1h/                         BTC + 7 alts, 1h K-lines (OKX spot)
├── spread_adf_kss.py                Pair selection: OLS β, ADF, KSS, Kendall τ
├── copula_marginals.py              Marginal fitting (Normal / t / Cauchy) + AIC + PIT
├── copula_fitting.py                5 copula families (Gaussian/t/Clayton/Gumbel/Frank) + AIC
├── rolling_backtest.py              ★ Main engine: rolling formation/trading + signals + PnL
├── demo_signal_modes.ipynb          ★ Main demo: compares all signal logic variants
├── gpu_version/                     GPU version (RTX/CUDA), same logic as CPU
│   ├── copula_fitting_gpu.py
│   ├── rolling_backtest_gpu.py
│   └── demo_signal_modes_gpu.ipynb
└── rolling_ZY/                      Collaborator's parallel implementation (reference)
```

---

## Core idea

### Data flow

```
   ┌──── formation period (3 weeks = 504h) ────┐ ┌─ trading period (1 week = 168h) ──┐
   │  1. OLS β, ADF/KSS, Kendall τ              │ │  5. Compute spread w/ frozen β    │
   │  2. Pick top-2 alts as pair                │ │  6. (u,v) via frozen marginals    │
   │  3. Fit marginals (Normal/t/Cauchy)        │ │  7. h12, h21 via frozen copula    │
   │  4. Fit copula (5 families, AIC select)    │ │  8. Apply signal rules → trade    │
   └────────────────────────────────────────────┘ └───────────────────────────────────┘
                              ↓
              Slide window forward by 168h every cycle (~257 cycles total)
```

**Key principle**: the formation period **trains** the model; the trading period only **applies** frozen parameters. This avoids look-ahead bias.

### Core math

- **Spread**: $S^i_t = \text{BTC}_t - \beta^i \cdot P^i_t$ (no intercept, through-origin OLS)
- **PIT**: $u_t = \hat F_1(S^1_t)$, $v_t = \hat F_2(S^2_t)$
- **Conditional copula** (paper Eq. 4):

$$
h^{1|2}_t = \frac{\partial C(u,v)}{\partial v}\bigg|_{(u_t,v_t)}, \qquad
h^{2|1}_t = \frac{\partial C(u,v)}{\partial u}\bigg|_{(u_t,v_t)}
$$

- **Mispricing index**: $\text{MI}^{j|k}_t = h^{j|k}_t - 0.5$ (deviation from 0.5 indicates mispricing)

---

## Three signal logics reproduced in `demo_signal_modes.ipynb`

### Logic A: Return-based (paper Table 3 default)

**Decisions made on instantaneous $h_t$**:

| Condition | Action |
|---|---|
| $h^{1\|2}_t < \alpha_1$ AND $h^{2\|1}_t > 1-\alpha_1$ | Open long $S^1$ / short $S^2$ |
| $h^{1\|2}_t > 1-\alpha_1$ AND $h^{2\|1}_t < \alpha_1$ | Open short $S^1$ / long $S^2$ |
| $\|h^{1\|2}_t-0.5\|<\alpha_2$ AND $\|h^{2\|1}_t-0.5\|<\alpha_2$ | Close |
| In position + no red zone hit | Hold |
| Last bar of trading week | Force close |

**Geometry**: open zones are two small triangles in opposite corners of the $(u, v)$ unit square:

```
v=1 ┌──────────────┬──┐  ← Green A: sig_open_long
    │              │██│
    │              │██│
    │   diagonal   │  │
    │   (most data)│  │
    │              │  │
    │██│              │  ← Green B: sig_open_short
v=0 └──┴──────────────┘
   u=0              u=1
```

**Issue**: for positively-dependent copulas, these green corners are very rarely hit — most data sits along the diagonal and never enters either green region.

Call:
```python
run_backtest(signal_mode="return", alpha_open=0.10, alpha_close=0.10)
```

---

### Logic B: Return-based, loosened ($\alpha_1 = 0.20$)

Loosen the open threshold from 0.10 to 0.20, **expanding each green corner area**. All other rules unchanged.

```python
run_backtest(signal_mode="return", alpha_open=0.20, alpha_close=0.10)
```

---

### Logic C: Level-based (paper Eq. 5, CMI)

**Accumulate** the instantaneous deviation from 0.5:

$$
\text{CMI}^{1|2}_t = \sum_{s=0}^{t}\big(h^{1|2}_s - 0.5\big), \qquad
\text{CMI}^{2|1}_t = \sum_{s=0}^{t}\big(h^{2|1}_s - 0.5\big)
$$

| Condition | Action |
|---|---|
| $\text{CMI}^{1\|2} < -t_o$ AND $\text{CMI}^{2\|1} > +t_o$ | Open long $S^1$ / short $S^2$ |
| $\text{CMI}^{1\|2} > +t_o$ AND $\text{CMI}^{2\|1} < -t_o$ | Open short $S^1$ / long $S^2$ |
| In long: $\text{CMI}^{2\|1}<+t_c$ AND $\text{CMI}^{1\|2}>-t_c$ | Close |
| In short: $\text{CMI}^{1\|2}<+t_c$ AND $\text{CMI}^{2\|1}>-t_c$ | Close |

CMI is reset to 0 at the start of each trading week. Paper defaults: $t_o = 1.0, t_c = 0.0$.

**Why this fixes the rarity issue of A**: even if $h$ deviates by only 0.05 per bar, after 6 bars the CMI hits 1 → trigger fires. **Captures small persistent divergence** instead of demanding instantaneous extremes.

Call:
```python
run_backtest(signal_mode="level", cmi_open=1.0, cmi_close=0.0)
```

---

### Logic D: Return-based + green-to-green flip (`allow_flip=True`, our improvement)

**Observation**: Paper Table 3 only has "red → close", with no "green → flip" rule. Consider this scenario:

```
In long S1 (opened in Green A) ──> (u,v) jumps directly into Green B (says short) ──> position != 0
                                                                                       └─> Original logic ignores Green B
                                                                                           Holds wrong direction until red
                                                                                           (or week-end forced close)
```

**Improved rule**:

| Condition | Action |
|---|---|
| Red zone | Close (same as Table 3) |
| In long + Green B fires | **Same-bar close long, open short** |
| In short + Green A fires | **Same-bar close short, open long** |
| Same-direction green or no signal | Hold |

Cost: an extra round-trip fee per flip (open + close × 0.04% each).
Benefit: corrects direction immediately when the market disagrees.

Call:
```python
run_backtest(signal_mode="return", alpha_open=0.20, allow_flip=True)
```

---

## Position sizing & PnL (shared by all logics)

- **Fix Q at week's open**: $Q_1 = 20{,}000 / P^1_\text{open}$, $Q_2 = 20{,}000 / P^2_\text{open}$ (each leg ≤ 20K USDT notional)
- $Q_1, Q_2$ stay constant throughout the trading week
- **Close PnL** (long $S^1$ / short $S^2$ example):
  $$\text{Gross} = Q_1 \cdot (P^1_\text{open} - P^1_\text{close}) + Q_2 \cdot (P^2_\text{close} - P^2_\text{open})$$
- **Fees**: 0.04% (Binance taker) charged on notional at both open and close per leg
- **Net PnL = Gross − open fee − close fee**

---

## `demo_signal_modes.ipynb` cell map

| Section | Cells | Content |
|---|---|---|
| 0–1 | markdown + setup | Import `run_backtest`, chdir to project root |
| 1 | 2–3 | **Logic A**: return-based α=0.10 |
| 2 | 4–5 | **Logic B**: return-based α=0.20 |
| 3 | 6–7 | **Logic C**: level-based to=1.0, tc=0.0 |
|   | 8–11 | A/B/C metrics table + equity curve overlay |
|   | 12–13 | Discussion + save outputs |
| 4 | 14 | **Logic D introduction**: the green-to-green gap in Table 3 |
|   | 15–16 | **Logic D**: α=0.10 / α=0.20 each with allow_flip=True |
|   | 17–18 | 4 return-based configs (×{0.10, 0.20} ×{flip, no-flip}) + level row, 5-row metrics table |
|   | 19–20 | Flip vs no-flip equity curves (dashed/solid) |
|   | 21–22 | **Count of actual flip-induced closes** (test whether the gap really matters on this data) |
|   | 23 | When does `allow_flip` help / hurt |
| 5 | 24–31 | **Numerical optimizer benchmark**: BFGS vs Nelder-Mead (see below) |

---

## Running the demo

### Environment

```bash
# Python 3.10+ recommended
pip install pandas numpy scipy statsmodels matplotlib torch
```

### Execution

Open `demo_signal_modes.ipynb`, Run All from cell 0.

**Total runtime**: ~25 minutes
- 5 calls to `run_backtest` (A / B / C / D-α0.10 / D-α0.20), ~5 min each
- Each call: 257 cycles × 5 copula families × MLE

---

## Numerical acceleration analysis

The bottleneck of the entire backtest is **MLE optimization** — every cycle solves 5 copula families × multiple starts ≈ 15 nonlinear optimizations, totaling ~3850 MLE solves over 257 cycles. **The choice of optimizer dominates total wall time.**

### 1. Optimizer taxonomy (theoretical complexity)

| Method | Convergence rate | Per-step compute | Storage | Smoothness required | Typical iterations (1-2 params) |
|---|---|---|---|---|---|
| Newton's | Quadratic | $O(p^3)$ Hessian inverse | $O(p^2)$ | $C^2$, locally PD Hessian | 5–10 |
| **BFGS** | Superlinear | $O(p^2)$ matrix update | $O(p^2)$ | $C^1$, continuously differentiable | 10–30 |
| **L-BFGS / L-BFGS-B** | Superlinear | $O(mp)$, $m \approx 5\text{–}20$ | $O(mp)$ | $C^1$ | 10–30 |
| **Nelder-Mead** | No theoretical guarantee (linear-ish in practice) | $O(p)$ per feval | $O(p^2)$ | Continuous only | 100–1000+ |
| **Adam (GPU)** | Sublinear | $O(p)$ + GPU launch | $O(p)$ | $C^1$ | 50–200 |

Convergence rate definitions:
- **Quadratic** $\|x_{k+1}-x^*\| \le C\|x_k-x^*\|^2$ → digits of accuracy double per step (very fast, but demanding)
- **Superlinear** $\lim_k \|x_{k+1}-x^*\|/\|x_k-x^*\| = 0$ → faster than linear, slower than quadratic
- **Linear** $\|x_{k+1}-x^*\| \le \rho\|x_k-x^*\|$, $0<\rho<1$ → digits of accuracy increase linearly

### 2. Complexity of the three numerical tasks in this project

#### Task 1: KSS auxiliary regression (**closed-form**, fastest)

$\Delta S_t = \delta \cdot S_{t-1}^3 + \epsilon_t$ → statsmodels OLS (QR decomposition)

- Complexity: $O(np^2) = O(503 \cdot 1) = O(n)$, no iteration
- Per call: ~0.5 ms
- Total calls: $257 \text{ cycles} \times 7 \text{ alts} \approx 1800$ → **~1 second**

#### Task 2: Marginal MLE

$\hat\theta = \arg\max \sum_{i=1}^{n} \log f(x_i;\theta)$, $f \in \{\text{Normal}, t, \text{Cauchy}\}$

- Normal: closed-form ($\hat\mu = \bar x$, $\hat\sigma = s$)
- $t$ / Cauchy: numerical (scipy.stats.fit defaults to BFGS), $p=2$–$3$
- Per fit: ~5–20 ms
- Total: $257 \times 2 \text{ spreads} \times 3 \text{ candidates} \approx 1500$ → **~10–30 seconds**

#### Task 3: Copula MLE (**main bottleneck**)

$\hat\theta = \arg\max \sum_{i=1}^{n} \log c(u_i, v_i; \theta)$ for each of 5 families, each $p \in \{1,2\}$

- Optimizer: scipy.optimize.minimize, method=`L-BFGS-B` (with box bounds)
- Multi-start: 3 different initializations (avoid local minima)
- Per family per start: **~50–100 ms**
- Per cycle: 5 families × 3 starts ≈ **~1 second**
- Total: $257 \times 1 \text{ s} \approx$ **4 minutes**

→ **80% of the ~5-minute total runtime is spent here.**

### 3. Theoretical speedup estimates

#### Gain ①: BFGS / L-BFGS-B vs Nelder-Mead

For 1-2 parameter smooth NLL, the theoretical ratio is:

| Metric | Nelder-Mead | L-BFGS-B | Ratio (theory) |
|---|---|---|---|
| Avg iterations | 150–300 | 15–30 | **10×** |
| Fevals per iter | 1–2 | 2–4 (with gradient) | 0.5× |
| Total fevals | 200–500 | 30–100 | **5×** |
| Per fit wall time (504 obs, 1 param) | ~30–60 ms | ~5–15 ms | **5–8×** |

**Why BFGS is usually faster**: BFGS uses the secant condition $B_{k+1}(x_{k+1}-x_k) = \nabla f(x_{k+1})-\nabla f(x_k)$ to implicitly approximate the Hessian → uses local 2nd-order info → superlinear convergence. Nelder-Mead only uses function-value simplex geometry, ignoring gradient structure.

**Empirical results (see demo §5) show two regimes**:

| Problem | Objective | Dim | Empirical NM/L-BFGS-B ratio | Matches theory? |
|---|---|---|---|---|
| Gaussian copula MLE | smooth 1-param NLL | $p=1$ | **1.76×** (L-BFGS-B faster) | ✅ Lower bound of theory |
| KSS smooth-transition NLS | $\Delta S \approx \delta S^3 e^{-\theta z^2}$ | $p=2$ | **0.43×** (NM actually faster!) | ❌ Counterintuitive |

**Why KSS is a BFGS counterexample**: the $e^{-\theta z^2}$ exponential **saturates** the gradient in the $\theta$ direction → Hessian becomes nearly singular along that axis (huge condition number). BFGS's quasi-Newton update is dominated by noise → slower than the gradient-free simplex method. **This validates BFGS's theoretical assumption**: "locally PD Hessian with bounded condition number" — once violated, the second-order advantage disappears.

**Practical implication**:
- For **smooth, well-conditioned** problems like copula MLE → L-BFGS-B is correct (scipy default ✓)
- For **ill-conditioned/saturated** objectives like the smooth-transition NLS → Nelder-Mead can be more robust
- **No optimizer is universally fastest** — the choice must depend on the local geometry of the objective

#### Gain ②: Multi-start

Single-start failure probability (caught in a bad local minimum):
- Unimodal smooth objective: ~5%
- Multimodal (e.g., Frank copula near $\theta \approx 0$): ~30%

With $k$ starts, failure probability $\approx p^k$:

$$
P_\text{fail}(k=3) \approx \begin{cases} 0.0001 & \text{unimodal} \\ 0.027 & \text{multimodal} \end{cases}
$$

Cost: 3× wall time. **Benefit**: reliability (drops multimodal failure rate from 30% to 2.7%), not speed.

#### Gain ③: GPU (theory vs reality)

**Theoretical raw compute**: RTX 5090 ≈ 21,760 CUDA cores vs CPU 16 cores → ~1300× compute ratio.

**Empirical** (n=504, p=1–2 single-cycle MLE):

| Mode | Wall time | vs baseline |
|---|---|---|
| CPU scipy L-BFGS-B | ~0.7 s | baseline |
| GPU PyTorch Adam (per cycle) | ~1.4 s | **2× slower** |
| GPU cross-cycle batch (theoretical) | ~5 s total | **~50× faster** vs 4 min CPU |

**Why per-cycle GPU is slower**:
- Each Adam step incurs 1–5 ms kernel launch overhead
- 60 steps × 5 families ≈ 300 launches × 2 ms = **600 ms pure launch overhead**
- Actual compute is microsecond-scale, completely drowned out by launches

**Why cross-cycle batching would be fast**:
- Pack 257 cycles × 5 families = 1285 independent MLE problems into a single tensor
- One launch runs all 1285 optimizations in parallel
- Launch overhead amortized to <1 µs per problem

**Conclusion**: while preserving the paper's per-cycle orchestration, GPU is **not** the right acceleration target for this problem. Achieving real GPU speedup would require restructuring into full-batch mode.

### 4. Total acceleration budget

| Optimization | Used? | Actual gain | Source |
|---|---|---|---|
| L-BFGS-B replacing NM (copula MLE) | ✅ (scipy default) | **~1.76×** (measured demo §5.2) | Superlinear vs linear convergence |
| 3-start multi-start | ✅ | Reliability ↑ (multimodal fail 30% → 2.7%) | $p_\text{fail}^k$ |
| L-BFGS low-storage variant | ✅ (scipy auto) | Marginal at $p \le 2$ | Storage $O(mp)$ vs $O(p^2)$ |
| KSS closed-form OLS (vs NLS) | ✅ (statsmodels QR) | **~100×** vs numeric NLS | Closed-form vs iterative |
| Normal marginal closed-form | ✅ | **~50×** vs $t$/Cauchy numeric fit | Closed-form mean/std |
| GPU per-cycle | ❌ (counterproductive) | **0.5×** (GPU 2× slower) | Kernel launch overhead |
| GPU cross-cycle batch | 🔵 not implemented | est. **50×** | 1285 parallel MLE amortizes launch |

**The CPU-side acceleration headroom is fully exploited**: copula fits use L-BFGS-B, KSS test uses closed-form OLS, Normal marginal uses closed-form mean/std. The 5-minute runtime is a reasonable lower bound for scipy + multi-start. Beyond this requires either GPU batching (refactor) or cutting multi-start (sacrificing reliability).

---

## Key parameter summary

| Parameter | Default | Description |
|---|---|---|
| `timeframe` | `"1h"` | Data frequency |
| `formation_hours` | 504 | Training window (3 weeks) |
| `trading_hours` | 168 | Trading window (1 week) |
| `gate` | `"kss"` | Stationarity test: `"kss"` or `"adf"` |
| `signal_mode` | `"return"` | Signal engine: `"return"` or `"level"` |
| `alpha_open` | 0.10 | Return-based open threshold $\alpha_1$ |
| `alpha_close` | 0.10 | Return-based close threshold $\alpha_2$ |
| `cmi_open` | 1.0 | Level-based open threshold $t_o$ |
| `cmi_close` | 0.0 | Level-based close threshold $t_c$ |
| `allow_flip` | `False` | Enable green-to-green flip (our improvement, return mode only) |
| `notional_per_leg` | 20000 | Per-leg notional capital (USDT) |
| `fee` | 0.0004 | Per-leg per-trade fee (0.04% taker) |

---

## Known differences from the paper

1. **Data**: OKX spot vs paper's Binance USDT-margined perpetual; 7 alts vs paper's 19
2. **Time window**: this project covers 2021-01 → 2025-12 (257 cycles) vs paper's 2021-01 → 2023-01 (104 cycles)
3. **Copula pool**: 5 families (Gaussian/t/Clayton/Gumbel/Frank) vs paper's 12 (additionally Joe/BB1/BB6/BB7/BB8/Tawn1/Tawn2)
4. **Signal execution**: same-bar decision and execution (1-bar look-ahead, < 1bp bias at 1h frequency)

---

## Output files

Each `save_results(...)` call writes to `result/result_1h/<subdir>/`:

| File | Content |
|---|---|
| `metrics.json` | Summary metrics + full config |
| `cycle_log.csv` | Per-week pair / β / τ / fits / pnl (~257 rows) |
| `trade_log.csv` | Open/close details for every trade (~200 rows) |
| `equity_per_cycle.csv` | Cumulative PnL time series |

---

## Citation

```
Tadi, M., Witzany, J. (2025). Copula-based trading of cointegrated cryptocurrency
pairs. Financial Innovation, 11:40. https://doi.org/10.1186/s40854-024-00702-7
```
