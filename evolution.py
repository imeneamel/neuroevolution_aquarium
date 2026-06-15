"""
Aquarium Evolution Engine — v6 (miroir de ab.js)
=================================================
Paradigme : population ouverte, reproduction sexuée, tempérament héréditaire.
Plus de "lignées figées" ni de JSON à charger dans le HTML.
Ce script est un outil d'ANALYSE HEADLESS :
  - Simule N_RUNS populations indépendantes sur SIM_TICKS ticks
  - Mesure convergence des tempéraments, fitness, diversité génétique
  - Exporte stats.json (optionnel) et affiche les résultats en console

Usage :
    python evolution.py              # analyse standard
    python evolution.py --ticks 8000 --runs 3
"""

import argparse, json, math, random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════
#  Constantes — miroir EXACT de ab.js
# ══════════════════════════════════════════════════════════════
W, H = 800, 600

# MLP
INPUT_SIZE  = 16
HIDDEN_SIZE = 18
OUTPUT_SIZE = 4
N_W1 = INPUT_SIZE  * HIDDEN_SIZE
N_B1 = HIDDEN_SIZE
N_W2 = HIDDEN_SIZE * OUTPUT_SIZE
N_B2 = OUTPUT_SIZE
N_WEIGHTS = N_W1 + N_B1 + N_W2 + N_B2

# Physique
MAX_SPD        = 3.2
FISH_SPD       = 3.0
PANIC_DIST_BASE= 100.0
FOOD_R         = 11
PRED_R         = 16
FISH_R         = 9
BORDER_ZONE    = 80.0
WALL_FORCE     = 2.6
WALL_EXP       = 1.6
MEAL_SAT_TICKS = 400.0
FOOD_MIN_DIST  = 28

# Écologie
N_FOOD             = 34
FOOD_RESPAWN_BASE  = 0.045
ECO_K              = 9
HUNGER_GAIN        = 0.40

# Cycle de vie
START_POP      = 9
MIN_POP        = 3
MATURITY_AGE   = 600
REPRO_ENERGY   = 0.72
REPRO_COST     = 0.30
REPRO_COOLDOWN = 500
REPRO_RANGE    = 42
MAX_AGE        = 9000
OLD_AGE_DEATH  = 0.0006

# Génétique
MUT_STD          = 0.12
MUT_BIG_PROB     = 0.05
MUT_BIG_MULT     = 3.0
TEMP_MUT_STD     = 0.06

# Prédateur
AGGRO_R          = 230.0
ABANDON_R        = 290.0
MAX_CHASE_TICKS  = 220
PRED_BASE_SPEED  = 1.6

# ══════════════════════════════════════════════════════════════
#  MLP — miroir exact de ab.js (layout poids identique)
# ══════════════════════════════════════════════════════════════
def xavier(fan_in, fan_out, n, rng):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, n)

def random_weights(rng) -> np.ndarray:
    w = np.zeros(N_WEIGHTS)
    idx = 0
    w[idx:idx+N_W1] = xavier(INPUT_SIZE, HIDDEN_SIZE, N_W1, rng); idx += N_W1
    idx += N_B1   # biais initialisés à 0
    w[idx:idx+N_W2] = xavier(HIDDEN_SIZE, OUTPUT_SIZE, N_W2, rng); idx += N_W2
    # b2 = 0
    return w

def mlp_forward(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    idx = 0
    W1 = weights[idx:idx+N_W1].reshape(HIDDEN_SIZE, INPUT_SIZE);  idx += N_W1
    b1 = weights[idx:idx+N_B1];                                    idx += N_B1
    W2 = weights[idx:idx+N_W2].reshape(OUTPUT_SIZE, HIDDEN_SIZE);  idx += N_W2
    b2 = weights[idx:idx+N_B2]
    h  = np.tanh(W1 @ x + b1)
    return np.tanh(W2 @ h + b2)

def crossover_weights(wa: np.ndarray, wb: np.ndarray, rng) -> np.ndarray:
    a = 0.25 + rng.random() * 0.5   # a in [0.25, 0.75]
    return a * wa + (1 - a) * wb

def mutate_weights(w: np.ndarray, rng) -> np.ndarray:
    noise = rng.normal(0, MUT_STD, N_WEIGHTS)
    big   = rng.random(N_WEIGHTS) < MUT_BIG_PROB
    noise[big] *= MUT_BIG_MULT
    return w + noise

# ══════════════════════════════════════════════════════════════
#  Tempérament — miroir de ab.js
# ══════════════════════════════════════════════════════════════
TEMP_KEYS = ('perception', 'metabolism', 'socialPull', 'boldness')

def random_temperament(rng) -> dict:
    return {k: float(rng.random()) for k in TEMP_KEYS}

def crossover_temperament(ta: dict, tb: dict, rng) -> dict:
    out = {}
    for k in TEMP_KEYS:
        a = rng.random()
        out[k] = float(np.clip(a*ta[k] + (1-a)*tb[k], 0, 1))
    return out

def mutate_temperament(t: dict, rng) -> dict:
    return {k: float(np.clip(t[k] + rng.normal(0, TEMP_MUT_STD), 0, 1)) for k in TEMP_KEYS}

def temperament_label(t: dict) -> str:
    tags = []
    tags.append('myope'       if t['perception'] < 0.35 else 'clairvoyant' if t['perception'] > 0.65 else None)
    tags.append('économe'     if t['metabolism'] < 0.35 else 'glouton'     if t['metabolism'] > 0.65 else None)
    tags.append('solitaire'   if t['socialPull'] < 0.35 else 'grégaire'    if t['socialPull'] > 0.65 else None)
    tags.append('anxieux'     if t['boldness']   < 0.35 else 'téméraire'   if t['boldness']   > 0.65 else None)
    present = [x for x in tags if x]
    return ' · '.join(present) if present else 'équilibré'

# ══════════════════════════════════════════════════════════════
#  Entités
# ══════════════════════════════════════════════════════════════
_next_id = 1

@dataclass
class Fish:
    id:           int
    gen:          int
    weights:      np.ndarray
    temperament:  dict
    x: float;    y: float
    vx: float = 0.0;  vy: float = 0.0
    alive:        bool = True
    age:          int  = 0
    energy:       float = 1.0
    food_eaten:   int  = 0
    steps_survived: int = 0
    distance_traveled: float = 0.0
    fear:         float = 0.0
    fear_accum:   float = 0.0
    fear_avg:     float = 0.0
    danger_mem:   float = 0.0
    time_since_meal: float = MEAL_SAT_TICKS * 0.4
    meal_intervals: list = field(default_factory=list)
    repro_cooldown: float = MATURITY_AGE * 0.3
    children:     int  = 0
    death_cause:  Optional[str] = None
    parents:      Optional[tuple] = None

@dataclass
class Food:
    x: float; y: float
    eaten: bool = False

@dataclass
class Predator:
    x: float; y: float
    vx: float = 0.0; vy: float = 0.0
    active: bool = False
    on_timer: int = 0
    cooldown: float = 0.0
    mode: str = 'rôde'
    chase_ticks: int = 0
    ambush_ticks: int = 0
    ambush_x: float = W/2; ambush_y: float = H/2
    prev_tx: Optional[float] = None; prev_ty: Optional[float] = None
    speed: float = PRED_BASE_SPEED
    target_id: Optional[int] = None

# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════
def d(x1,y1,x2,y2): return math.hypot(x2-x1, y2-y1)

def place_food_no_overlap(f: Food, pool: list, rng):
    for _ in range(8):
        nx = 30 + rng.random()*(W-60)
        ny = 30 + rng.random()*(H-100)
        ok = all(g is f or g.eaten or d(nx,ny,g.x,g.y) >= FOOD_MIN_DIST for g in pool)
        if ok: f.x, f.y = nx, ny; return
    f.x, f.y = 30 + rng.random()*(W-60), 30 + rng.random()*(H-100)

def make_food(pool, rng) -> Food:
    f = Food(0, 0)
    place_food_no_overlap(f, pool, rng)
    return f

def make_fish(rng, gen=0, weights=None, temperament=None,
              x=None, y=None, energy=1.0, repro_cooldown=None, parents=None) -> Fish:
    global _next_id
    t = temperament or random_temperament(rng)
    fish = Fish(
        id=_next_id, gen=gen,
        weights=weights if weights is not None else random_weights(rng),
        temperament=t,
        x=x if x is not None else 60+rng.random()*(W-120),
        y=y if y is not None else 60+rng.random()*(H-120),
        energy=energy,
        repro_cooldown=repro_cooldown if repro_cooldown is not None else MATURITY_AGE*0.3,
        parents=parents,
    )
    _next_id += 1
    return fish

def wall_rep(pos, lo, hi, zone=BORDER_ZONE, force=WALL_FORCE, exp=WALL_EXP):
    r = 0.0
    dlo = pos - lo; dhi = hi - pos
    if dlo < zone: r += force*(1-dlo/zone)**exp
    if dhi < zone: r -= force*(1-dhi/zone)**exp
    return r

# ══════════════════════════════════════════════════════════════
#  Prédateur — miroir de stepPredator() dans ab.js
# ══════════════════════════════════════════════════════════════
def step_predator(pred: Predator, alive_fishes: list, rng):
    # Timer
    if pred.active:
        pred.on_timer -= 1
        if pred.on_timer <= 0:
            pred.active    = False
            pred.cooldown  = 280 + rng.random()*200
            pred.mode      = 'rôde'
            pred.chase_ticks = 0
            pred.prev_tx   = pred.prev_ty = None
            pred.target_id = None
    else:
        pred.cooldown -= 1
        if pred.cooldown <= 0:
            pred.active       = True
            pred.on_timer     = int(160 + rng.random()*100)
            pred.chase_ticks  = pred.ambush_ticks = 0
            pred.prev_tx      = pred.prev_ty = None
            pred.mode         = 'rôde'
            side = rng.integers(0, 4)
            if   side == 0: pred.x, pred.y = 8,   60+rng.random()*(H-120)
            elif side == 1: pred.x, pred.y = W-8,  60+rng.random()*(H-120)
            elif side == 2: pred.x, pred.y = 60+rng.random()*(W-120), 8
            else:           pred.x, pred.y = 60+rng.random()*(W-120), H-8
            pred.vx = pred.vy = 0.0

    if not pred.active or not alive_fishes:
        pred.vx = pred.vx*0.93 + (W/2-pred.x)/W*0.5 + (rng.random()-0.5)*0.6
        pred.vy = pred.vy*0.93 + (H/2-pred.y)/H*0.5 + (rng.random()-0.5)*0.6
        pred.mode = 'rôde'
        pred.x = float(np.clip(pred.x+pred.vx, 5, W-5))
        pred.y = float(np.clip(pred.y+pred.vy, 5, H-5))
        return

    # Vitesse adaptée à l'audace moyenne (miroir ab.js)
    avg_bold = sum(f.temperament['boldness'] for f in alive_fishes) / len(alive_fishes)
    target_spd = PRED_BASE_SPEED * (0.85 + avg_bold*0.35)
    pred.speed = pred.speed*0.95 + target_spd*0.05

    # Cible avec hystérésis
    nearest = min(alive_fishes, key=lambda f: d(pred.x,pred.y,f.x,f.y))
    min_d   = d(pred.x, pred.y, nearest.x, nearest.y)
    target_fish = next((f for f in alive_fishes if f.id == pred.target_id), None)
    if target_fish is None:
        target_fish = nearest; pred.prev_tx = pred.prev_ty = None
    else:
        d_cur = d(pred.x,pred.y,target_fish.x,target_fish.y)
        if nearest is not target_fish and min_d < d_cur*0.6:
            target_fish = nearest; pred.prev_tx = pred.prev_ty = None
        else:
            min_d = d_cur
    pred.target_id = target_fish.id

    if min_d < AGGRO_R:
        pred.mode = 'chasse'; pred.chase_ticks += 1; pred.ambush_ticks = 0
        tx, ty = target_fish.x, target_fish.y
        if pred.prev_tx is not None:
            h = min(min_d / max(pred.speed*1.5, 0.1), 18)
            tx = float(np.clip(tx + (tx-pred.prev_tx)*h, 5, W-5))
            ty = float(np.clip(ty + (ty-pred.prev_ty)*h, 5, H-5))
        pred.prev_tx, pred.prev_ty = target_fish.x, target_fish.y
        dx = tx-pred.x; dy = ty-pred.y; dn = max(math.hypot(dx,dy), 1)
        pred.vx = pred.vx*0.50 + (dx/dn)*pred.speed*0.50
        pred.vy = pred.vy*0.50 + (dy/dn)*pred.speed*0.50

        if min_d > ABANDON_R or pred.chase_ticks > MAX_CHASE_TICKS:
            pred.mode = 'embuscade'; pred.chase_ticks = 0
            pred.prev_tx = pred.prev_ty = None
            pcx = sum(f.x for f in alive_fishes)/len(alive_fishes)
            pcy = sum(f.y for f in alive_fishes)/len(alive_fishes)
            pred.ambush_x = (pred.x+pcx)/2; pred.ambush_y = (pred.y+pcy)/2
            pred.ambush_ticks = 0

    elif pred.mode == 'embuscade':
        pred.ambush_ticks += 1
        dx = pred.ambush_x-pred.x; dy = pred.ambush_y-pred.y
        dn = max(math.hypot(dx,dy), 1)
        if dn > 10:
            pred.vx = pred.vx*0.7 + (dx/dn)*pred.speed*0.12
            pred.vy = pred.vy*0.7 + (dy/dn)*pred.speed*0.12
        else:
            pred.vx *= 0.8; pred.vy *= 0.8
        if pred.ambush_ticks > 90 + rng.random()*60:
            pred.mode = 'rôde'; pred.ambush_ticks = 0
            pred.prev_tx = pred.prev_ty = None
    else:
        pred.mode = 'rôde'; pred.chase_ticks = 0
        pred.prev_tx = pred.prev_ty = None
        pcx = sum(f.x for f in alive_fishes)/len(alive_fishes)
        pcy = sum(f.y for f in alive_fishes)/len(alive_fishes)
        dx = pcx-pred.x; dy = pcy-pred.y; dn = max(math.hypot(dx,dy), 1)
        pred.vx = pred.vx*0.92 + (dx/dn)*pred.speed*0.22 + (rng.random()-0.5)*0.3
        pred.vy = pred.vy*0.92 + (dy/dn)*pred.speed*0.22 + (rng.random()-0.5)*0.3

    pspd = math.hypot(pred.vx, pred.vy)
    if pspd > pred.speed*1.6:
        pred.vx *= pred.speed*1.6/pspd; pred.vy *= pred.speed*1.6/pspd
    pred.x = float(np.clip(pred.x+pred.vx, 5, W-5))
    pred.y = float(np.clip(pred.y+pred.vy, 5, H-5))

# ══════════════════════════════════════════════════════════════
#  Poisson — miroir de stepFish() dans ab.js
# ══════════════════════════════════════════════════════════════
def step_fish(fish: Fish, alive_fishes: list, foods: list, preds: list, rng):
    t = fish.temperament
    fish.steps_survived += 1
    fish.age += 1
    if fish.repro_cooldown > 0: fish.repro_cooldown -= 1

    # Métabolisme
    hunger_dec = (1.0/(2500*0.75)) * (0.7 + t['metabolism']*0.65)
    fish.energy = max(0.0, fish.energy - hunger_dec)
    if fish.energy <= 0:
        fish.alive      = False
        fish.death_cause= 'starvation'
        return

    alive_foods = [f for f in foods if not f.eaten]
    pnoise = (1 - t['perception']) * 0.35

    # Nourriture la plus proche
    fd_dx = fd_dy = 0.0; fd_dist = 1.0
    if alive_foods:
        closest = min(alive_foods, key=lambda f: d(fish.x,fish.y,f.x,f.y))
        min_df  = d(fish.x, fish.y, closest.x, closest.y)
        fd_dx   = (closest.x-fish.x)/W + (rng.random()-0.5)*pnoise
        fd_dy   = (closest.y-fish.y)/H + (rng.random()-0.5)*pnoise
        fd_dist = float(np.clip(min_df/(W*0.5) + (rng.random()-0.5)*pnoise*0.5, 0, 1))

    # Prédateur
    pd_dx = pd_dy = 0.0; pd_dist = 1.0; min_pd = float('inf'); closest_pred = None
    for p in preds:
        dd = d(fish.x, fish.y, p.x, p.y)
        if dd < min_pd: min_pd = dd; closest_pred = p
    is_pred_active = closest_pred is not None and closest_pred.active
    if closest_pred:
        pd_dx   = (closest_pred.x-fish.x)/W + (rng.random()-0.5)*pnoise
        pd_dy   = (closest_pred.y-fish.y)/H + (rng.random()-0.5)*pnoise
        pd_dist = float(np.clip(min_pd/(W*0.5) + (rng.random()-0.5)*pnoise*0.5, 0, 1))

    # Murs
    wx = min(fish.x, W-fish.x)/(W*0.5)
    wy = min(fish.y, H-fish.y)/(H*0.5)

    # Signal social (miroir exact ab.js)
    soc_dx = soc_dy = 0.0; neighbors = 0
    for o in alive_fishes:
        if o is fish: continue
        dd = d(fish.x, fish.y, o.x, o.y)
        if 0.001 < dd < 160:
            soc_dx += (o.x-fish.x)/dd; soc_dy += (o.y-fish.y)/dd; neighbors += 1
    soc_signal = 0.0
    if neighbors > 0:
        soc_dx /= neighbors; soc_dy /= neighbors
        soc_signal = math.hypot(soc_dx, soc_dy)
    soc_weight = 0.2 + t['socialPull']*0.8

    # Danger memory
    if is_pred_active and min_pd < 260:
        fish.danger_mem = min(1.0, fish.danger_mem + 0.18*(1-pd_dist))
    else:
        fish.danger_mem = max(0.0, fish.danger_mem*0.97)

    # Faim urgence + signal temporel
    hunger_urgency = max(0.0, 1.0 - fish.energy*2.5)
    fish.time_since_meal = min(MEAL_SAT_TICKS, fish.time_since_meal+1)
    meal_signal = fish.time_since_meal / MEAL_SAT_TICKS

    inp = np.array([
        fd_dx, fd_dy, fd_dist,
        pd_dx, pd_dy, pd_dist,
        wx, wy,
        fish.vx/MAX_SPD, fish.vy/MAX_SPD,
        fish.fear,
        1.0 if is_pred_active else 0.0,
        fish.danger_mem,
        hunger_urgency,
        meal_signal,
        soc_signal,   # input 15 = densité/force du groupe
    ], dtype=np.float64)

    out = mlp_forward(fish.weights, inp)
    ax = (out[3]-out[2])*FISH_SPD
    ay = (out[0]-out[1])*FISH_SPD

    # Influence sociale directe (miroir ab.js)
    if neighbors > 0:
        ax += soc_dx * soc_weight * 0.9
        ay += soc_dy * soc_weight * 0.9

    # Répulsion mur
    ax += wall_rep(fish.x, 5, W-5)
    ay += wall_rep(fish.y, 5, H-5)

    # Réflexe panique (modulé par boldness)
    panic_dist = PANIC_DIST_BASE * (1.3 - t['boldness']*0.6)
    if is_pred_active and min_pd < panic_dist:
        dxp = fish.x - closest_pred.x; dyp = fish.y - closest_pred.y
        dn  = max(math.hypot(dxp,dyp), 1)
        ax  = (dxp/dn)*FISH_SPD*1.6
        ay  = (dyp/dn)*FISH_SPD*1.6

    max_spd = MAX_SPD * (0.92 + t['metabolism']*0.16)
    fish.vx = fish.vx*0.50 + ax*0.50
    fish.vy = fish.vy*0.50 + ay*0.50
    spd = math.hypot(fish.vx, fish.vy)
    if spd > max_spd: fish.vx *= max_spd/spd; fish.vy *= max_spd/spd

    prev_x, prev_y = fish.x, fish.y
    fish.x = float(np.clip(fish.x+fish.vx, 5, W-5))
    fish.y = float(np.clip(fish.y+fish.vy, 5, H-5))
    moved  = math.hypot(fish.x-prev_x, fish.y-prev_y)
    fish.distance_traveled += moved

    # Manger
    hunger_gain = HUNGER_GAIN * (0.9 + t['metabolism']*0.25)
    for f in alive_foods:
        if not f.eaten and d(fish.x,fish.y,f.x,f.y) < FISH_R+FOOD_R:
            fish.food_eaten += 1
            fish.energy      = min(1.0, fish.energy + hunger_gain)
            fish.meal_intervals.append(fish.time_since_meal)
            if len(fish.meal_intervals) > 30: fish.meal_intervals.pop(0)
            fish.time_since_meal = 0.0
            f.eaten = True

    # Mort prédateur
    for pred in preds:
        if pred.active and d(fish.x,fish.y,pred.x,pred.y) < FISH_R+PRED_R:
            fish.alive      = False
            fish.death_cause= 'predator'
            return

    # Peur
    fear_gain = 0.28 * (1.4 - t['boldness']*0.8)
    if is_pred_active and min_pd < 200:
        fish.fear = min(1.0, fish.fear + fear_gain*(1-pd_dist))
    else:
        fish.fear = max(0.0, fish.fear - 0.04)
    fish.fear_accum += fish.fear
    fish.fear_avg    = fish.fear_accum / max(fish.steps_survived, 1)

# ══════════════════════════════════════════════════════════════
#  Reproduction — miroir de handleReproduction()
# ══════════════════════════════════════════════════════════════
def handle_reproduction(fishes: list, stats: dict, gen_counter: list, rng) -> list:
    candidates = [f for f in fishes if f.alive and f.age > MATURITY_AGE
                  and f.energy > REPRO_ENERGY and f.repro_cooldown <= 0]
    if len(candidates) < 2: return []

    paired = set(); new_fish = []
    for i, a in enumerate(candidates):
        if a.id in paired: continue
        for b in candidates[i+1:]:
            if b.id in paired: continue
            if d(a.x,a.y,b.x,b.y) < REPRO_RANGE:
                paired.add(a.id); paired.add(b.id)
                cw = mutate_weights(crossover_weights(a.weights, b.weights, rng), rng)
                ct = mutate_temperament(crossover_temperament(a.temperament, b.temperament, rng), rng)
                child_gen = max(a.gen, b.gen) + 1
                gen_counter[0] = max(gen_counter[0], child_gen)

                a.energy -= REPRO_COST; b.energy -= REPRO_COST
                a.repro_cooldown = REPRO_COOLDOWN; b.repro_cooldown = REPRO_COOLDOWN
                a.children += 1; b.children += 1

                cx = (a.x+b.x)/2 + (rng.random()-0.5)*10
                cy = (a.y+b.y)/2 + (rng.random()-0.5)*10
                child = make_fish(rng, gen=child_gen, weights=cw, temperament=ct,
                                  x=cx, y=cy, energy=0.55, parents=(a.id, b.id))
                new_fish.append(child)
                stats['births'] += 1
                break
    return new_fish

# ══════════════════════════════════════════════════════════════
#  Simulation complète d'une population sur N ticks
# ══════════════════════════════════════════════════════════════
def run_simulation(sim_ticks: int, seed: int, verbose: bool = False) -> dict:
    global _next_id
    _next_id = 1
    rng = np.random.default_rng(seed)

    fishes: List[Fish] = [make_fish(rng) for _ in range(START_POP)]
    foods:  List[Food] = []
    for _ in range(N_FOOD):
        f = Food(0,0)
        place_food_no_overlap(f, foods, rng)
        foods.append(f)

    pred = Predator(x=W*0.85, y=H*0.5, cooldown=120+rng.random()*100)
    preds = [pred]

    stats = {'births':0, 'deaths':{'starvation':0,'predator':0,'old_age':0}}
    gen_counter = [0]
    pop_history: List[Dict] = []

    for tick in range(1, sim_ticks+1):
        alive_fishes = [f for f in fishes if f.alive]
        pop = len(alive_fishes)

        # Respawn nourriture
        resp_p = FOOD_RESPAWN_BASE * (ECO_K / (ECO_K + pop))
        for f in foods:
            if f.eaten and rng.random() < resp_p:
                place_food_no_overlap(f, foods, rng)
                f.eaten = False

        # Prédateur
        step_predator(pred, alive_fishes, rng)

        # Poissons
        for fish in fishes:
            if fish.alive:
                step_fish(fish, alive_fishes, foods, preds, rng)

        # Reproduction
        new_fish = handle_reproduction(fishes, stats, gen_counter, rng)
        fishes.extend(new_fish)

        # Vieillesse
        for fish in fishes:
            if fish.alive and fish.age > MAX_AGE:
                p = OLD_AGE_DEATH * (1 + (fish.age-MAX_AGE)/MAX_AGE)
                if rng.random() < p:
                    fish.alive = False; fish.death_cause = 'old_age'
                    stats['deaths']['old_age'] += 1

        # Anti-extinction
        still_alive = sum(1 for f in fishes if f.alive)
        if still_alive < MIN_POP:
            fishes.append(make_fish(rng, gen=gen_counter[0]))
            stats['births'] += 1

        # Nettoyage
        if len(fishes) > 60:
            alive_l  = [f for f in fishes if f.alive]
            dead_l   = [f for f in fishes if not f.alive][-20:]
            fishes   = alive_l + dead_l

        # Historique
        if tick % 50 == 0:
            alive_l = [f for f in fishes if f.alive]
            avg_e   = sum(f.energy for f in alive_l)/max(len(alive_l),1)
            avg_gen = sum(f.gen    for f in alive_l)/max(len(alive_l),1)
            avg_t   = {k: sum(f.temperament[k] for f in alive_l)/max(len(alive_l),1)
                       for k in TEMP_KEYS}
            pop_history.append({'tick':tick,'pop':len(alive_l),'avgEnergy':avg_e,
                                 'avgGen':avg_gen,'avgTemp':avg_t})
            if verbose and tick % 500 == 0:
                print(f"  tick {tick:5d} | pop={len(alive_l):3d} gen={avg_gen:.1f} "
                      f"energy={avg_e:.2f} | "
                      f"perc={avg_t['perception']:.2f} metab={avg_t['metabolism']:.2f} "
                      f"soc={avg_t['socialPull']:.2f} bold={avg_t['boldness']:.2f}")

    alive_l = [f for f in fishes if f.alive]
    return {
        'seed':          seed,
        'final_pop':     len(alive_l),
        'max_gen':       gen_counter[0],
        'births':        stats['births'],
        'deaths':        stats['deaths'],
        'pop_history':   pop_history,
        'final_avg_temp': {k: sum(f.temperament[k] for f in alive_l)/max(len(alive_l),1)
                           for k in TEMP_KEYS} if alive_l else {},
        'final_avg_energy': sum(f.energy for f in alive_l)/max(len(alive_l),1),
    }

# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Aquarium Evolution — analyse headless')
    parser.add_argument('--ticks', type=int, default=4000,  help='Ticks par run (défaut 4000)')
    parser.add_argument('--runs',  type=int, default=2,     help='Nombre de runs indépendants (défaut 2)')
    parser.add_argument('--seed',  type=int, default=42,    help='Seed du premier run')
    parser.add_argument('--json',  type=str, default=None,  help='Exporter les stats dans un fichier JSON')
    args = parser.parse_args()

    print(f"Aquarium Evolution v6 — {args.runs} run(s) × {args.ticks} ticks")
    print(f"Architecture MLP : {INPUT_SIZE}→{HIDDEN_SIZE}→{OUTPUT_SIZE} ({N_WEIGHTS} poids)")
    print(f"Tempérament : {list(TEMP_KEYS)}\n")

    all_results = []
    for run_i in range(args.runs):
        seed = args.seed + run_i * 1337
        print(f"── Run {run_i+1}/{args.runs} (seed={seed}) {'─'*40}")
        result = run_simulation(args.ticks, seed=seed, verbose=True)
        all_results.append(result)
        ft = result['final_avg_temp']
        print(f"\n  → pop finale : {result['final_pop']} | "
              f"max génération : {result['max_gen']} | naissances : {result['births']}")
        print(f"     morts : faim={result['deaths']['starvation']} "
              f"prédateur={result['deaths']['predator']} vieillesse={result['deaths']['old_age']}")
        if ft:
            label = temperament_label(ft)
            print(f"     tempérament dominant : {label}")
            print(f"     → perception={ft['perception']:.3f}  "
                  f"metabolism={ft['metabolism']:.3f}  "
                  f"socialPull={ft['socialPull']:.3f}  "
                  f"boldness={ft['boldness']:.3f}")

    # Convergence inter-runs
    if args.runs > 1 and all(r['final_avg_temp'] for r in all_results):
        print(f"\n{'─'*55}")
        print("Convergence inter-runs (variance du tempérament final) :")
        for k in TEMP_KEYS:
            vals = [r['final_avg_temp'][k] for r in all_results]
            print(f"  {k:12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Stats exportées → {args.json}")

if __name__ == '__main__':
    main()