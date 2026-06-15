# Neuroevolution Aquarium

Browser-based neuroevolution simulation. Agents (fish) evolve foraging and 
predator-avoidance strategies entirely at runtime — no training data, no gradient descent.
Evolution happens through sexual reproduction, crossover, and mutation across 30+ generations.

**[▶ Live demo](https://imeneamel.github.io/aquarium-neuroevolution/aquarium.html)**

---

## What this is

Each fish carries:
- A **MLP (16 → 18 → 4)** controlling movement decisions
- A **temperament vector** of 4 heritable traits (perception, metabolism, social pull, boldness)
  that modulate both sensory inputs and physics independently of the network weights

Reproduction is sexual: two mature, nearby, well-fed fish produce an offspring via 
BLX-α crossover of both weight vectors and temperament, followed by Gaussian mutation 
with occasional large jumps (`MUT_BIG_PROB`).

Population pressure comes from:
- A single predator with 3 behavioral modes (chase / ambush / roam), whose speed 
  auto-adjusts to the population's average boldness
- Starvation (food respawn rate follows a Beverton-Holt curve dampened by population density)
- Old-age mortality beyond tick 9000

---

## Architecture

| Component | Detail |
|---|---|
| Network | MLP, tanh activations, He initialization |
| Inputs (16) | Food direction/distance, predator direction/distance, wall proximity, velocity, fear, danger memory, hunger urgency, meal signal, social cohesion |
| Outputs (4) | Δvx up/down/left/right |
| Crossover | BLX-α (α ∈ [0.25, 0.75]) on weight vectors |
| Mutation | Gaussian σ=0.12, with 5% probability of ×3 jump |
| Temperament | 4 traits in [0,1], crossover + Gaussian mutation σ=0.06 |

---

## Observed convergence (10 min run, ×10 speed)

After ~388k ticks and 475 births, population temperament converged toward:

| Trait | Converged value | Why |
|---|---|---|
| Metabolism | **économe (low)** | 80% of deaths were starvation → high metabolism eliminated |
| Boldness | **téméraire (high)** | Only 19% predator deaths → ignoring danger and eating more was net positive |
| Social pull | **solitaire (low)** | Food pool is fixed (34 units) → grouping creates direct competition |
| Perception | **myope (low)** | Likely genetic drift — perception noise had low selective cost |

---

## Known limitations

- **Permutation problem**: BLX-α crossover on MLP weights doesn't account for neuron 
  alignment between parents. NEAT-style historical markings would produce more viable 
  offspring. At this network size the effect is tolerable but visible as "lobotomized" 
  children with poor early behavior.
- **Temperament does most of the selective work**: with no gradient and small networks, 
  weights likely converge to "doesn't block survival" rather than fine-tuned strategies. 
  The temperament vector is the real unit of selection.
- `evolution.py` is a headless analysis tool for convergence statistics across multiple 
  runs, not a training script. The simulation is fully self-contained in `aquarium.html` + `ab.js`.

---

## Run locally

```bash
# No dependencies — open directly in browser
open aquarium.html
```

Use ×10 speed to observe meaningful generational convergence within minutes.
