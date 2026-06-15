"""
Aquarium Evolution Engine v5
==============================
Corrections vs v4 (qui causaient les poissons bloqués aux bords,
prédateurs incohérents, nourriture qui se superpose) :

PHYSIQUE / MURS
---------------
1. Répulsion de mur RADICALEMENT adoucie : force 5.5→2.6, exposant
   2.2→1.6, et surtout le signal de répulsion est maintenant CLAMPÉ
   et mélangé (pas juste additionné brut) pour ne plus écraser
   complètement la décision du réseau près des coins. Avant, dans un
   coin, ax et ay pouvaient atteindre ±5.5 chacun (>> FISH_SPD=3.0),
   noyant toute intention du réseau → le poisson restait "collé"
   en oscillant.
2. Pénalité de bord dans la fitness FORTEMENT augmentée et
   exponentielle avec la proximité réelle (pas juste un ratio de
   ticks) → pression sélective beaucoup plus nette pour quitter
   les bords.
3. Mirroir EXACT JS/Python de wall_repulse (même force, même zone,
   même exposant).

PRÉDATEUR
---------
4. La vitesse du prédateur dans la visualisation suit maintenant
   `pred_aggro` de la lignée du poisson actuellement pourchassé
   (mirroir fidèle de l'entraînement, où chaque lignée affronte un
   prédateur à SA vitesse d'agressivité).
5. Le prédateur peut maintenant cibler n'importe quel poisson vivant
   (pas seulement "le plus proche au moment du spawn") et change de
   cible si une proie plus proche apparaît à <0.6×distance actuelle
   (mirroir de la règle "abandon de cible").

NOURRITURE
----------
6. Pool de nourriture PROPRE À CHAQUE LIGNÉE, dimensionné selon son
   contexte d'entraînement (rich=32, normal=22, sparse=10) — un
   poisson "Vert" (sparse) ne voit que SES 10 nourritures, comme à
   l'entraînement. Évite la superposition visuelle générale et rend
   le comportement cohérent avec ce que le réseau a appris.
7. Anti-superposition : toute nouvelle position de nourriture
   (spawn ou recyclage) est tirée jusqu'à respecter une distance
   minimale (28px) avec les autres nourritures de SON pool.

RÉSEAU (inputs 15, INPUT_SIZE 14→15)
-------------------------------------
8. Nouvel input 14 : "time_since_meal" — temps écoulé (normalisé,
   sature à 1.0 après ~400 ticks) depuis le dernier repas. Permet au
   réseau d'apprendre une vraie dynamique de "pression de faim qui
   redescend quand on mange régulièrement" — signal complémentaire
   de hunger_urgency (qui ne réagit qu'en fin de réserve). Avec ce
   signal, on peut observer le réseau apprendre à anticiper la faim
   plutôt que d'y réagir trop tard.

FITNESS / PRESSION DE SURVIE
-----------------------------
9. Mourir de faim sans avoir JAMAIS mangé : pénalité encore plus
   sévère (×0.05 au lieu de ×0.10) — "ne pas comprendre qu'il faut
   manger" est désormais quasi-fatal pour le score.
10. Bonus de régularité alimentaire : récompense additionnelle basée
    sur le temps moyen entre deux repas (favorise une stratégie de
    nutrition régulière plutôt que famine puis festin).
11. ENTRAÎNEMENT PLUS POUSSÉ : N_GENERATIONS 22→34, POP_SIZE 24→28,
    N_SEEDS 4→5 — convergence plus robuste, comportements de survie
    plus nets observables à la fin.

OUTPUT : identique (4 sorties différentielles)
"""

import numpy as np
import json
import random
from multiprocessing import Pool, cpu_count

# ── Monde ─────────────────────────────────────────────────────────────────────
WORLD_W, WORLD_H = 800, 600
N_FOOD_BASE      = 22
MAX_STEPS        = 2500
N_GENERATIONS    = 34
POP_SIZE         = 28
ELITE_K          = 6
N_SEEDS          = 5
MUT_INIT         = 0.13
MUT_MIN          = 0.025
MUT_MAX          = 0.32
STAG_WINDOW      = 4

# ── Réseau (MIROIR EXACT DANS LE HTML) ────────────────────────────────────────
# Inputs (15) :
#  0-2  : nourriture  (dx/W, dy/H, dist_norm)
#  3-5  : prédateur   (dx/W, dy/H, dist_norm)
#  6-7  : murs        (dist_x_norm, dist_y_norm)
#  8-9  : vélocité    (vx/MAX_SPD, vy/MAX_SPD)
#  10   : peur        [0,1]
#  11   : pred_active [0,1]
#  12   : danger_memory [0,1]
#  13   : hunger_urgency [0,1]
#  14   : time_since_meal [0,1]  ← NOUVEAU
INPUT_SIZE  = 15
HIDDEN_SIZE = 18
OUTPUT_SIZE = 4
MAX_SPD     = 3.2
FISH_SPD    = 3.0
PANIC_DIST  = 100

HUNGER_DEC   = 1.0 / (MAX_STEPS * 0.75)   # meurt si 0 nourriture sur 75% de la durée
HUNGER_GAIN  = 0.40
BORDER_ZONE  = 80    # pixels : zone de répulsion / pénalité bord
WALL_FORCE   = 2.6   # force de répulsion (mirroir exact JS)
WALL_EXP     = 1.6   # exposant de répulsion (mirroir exact JS)
MEAL_SAT_TICKS = 400.0  # ticks pour saturer time_since_meal à 1.0

LINEAGE_CONTEXTS = {
    "Rouge":  {"pred_aggro": 2.6, "food": "normal", "color": "#e74c3c",
               "desc": "Prédateur ultra-agressif — sélection fuite pure"},
    "Bleu":   {"pred_aggro": 0.9, "food": "rich",   "color": "#3498db",
               "desc": "Env. favorable — optimise rendement alimentaire"},
    "Vert":   {"pred_aggro": 1.6, "food": "sparse", "color": "#27ae60",
               "desc": "Nourriture rare — exploration longue distance"},
    "Violet": {"pred_aggro": 2.0, "food": "normal", "color": "#8e44ad",
               "desc": "Pression mixte — compromis fuite/chasse"},
    "Orange": {"pred_aggro": 1.1, "food": "sparse", "color": "#e67e22",
               "desc": "Prédateur lent, env. pauvre — efficacité pure"},
}


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, w=None):
        if w is None:
            s1 = np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
            s2 = np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
            self.w = np.concatenate([
                np.random.randn(INPUT_SIZE * HIDDEN_SIZE) * s1,
                np.zeros(HIDDEN_SIZE),
                np.random.randn(HIDDEN_SIZE * OUTPUT_SIZE) * s2,
                np.zeros(OUTPUT_SIZE),
            ])
        else:
            self.w = w.copy()

    def forward(self, x):
        i = 0
        W1 = self.w[i:i+INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE); i += INPUT_SIZE*HIDDEN_SIZE
        b1 = self.w[i:i+HIDDEN_SIZE]; i += HIDDEN_SIZE
        W2 = self.w[i:i+HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE); i += HIDDEN_SIZE*OUTPUT_SIZE
        b2 = self.w[i:i+OUTPUT_SIZE]
        h  = np.tanh(x @ W1 + b1)
        return np.tanh(h @ W2 + b2)

    def mutate(self, std):
        # Mutation avec prob de perturbation forte occasionnelle (5%)
        noise = np.random.randn(*self.w.shape) * std
        big_mut_mask = np.random.rand(*self.w.shape) < 0.05
        noise[big_mut_mask] *= 3.0
        return MLP(self.w + noise)

    def crossover(self, other):
        # BLX-alpha crossover
        a = np.random.uniform(0.25, 0.75)
        return MLP(a * self.w + (1 - a) * other.w)

    def to_list(self):
        return self.w.tolist()

    @classmethod
    def from_list(cls, d):
        return cls(np.array(d))


# ── Prédateur (3 modes) ───────────────────────────────────────────────────────

class Pred:
    """
    Modes :
    - 'rôde'      : dérive lente vers le centroïde des proies
    - 'embuscade' : s'immobilise dans un point stratégique, attend
    - 'chasse'    : interception prédictive avec horizon adaptatif
    """
    AGGRO_R     = 230
    ABANDON_R   = 290
    MAX_CHASE   = 220
    AMBUSH_TICKS = 80   # durée min d'une embuscade

    def __init__(self, aggro_speed: float, rng: np.random.Generator):
        self.x = float(rng.uniform(100, WORLD_W - 100))
        self.y = float(rng.uniform(100, WORLD_H - 100))
        self.vx = self.vy = 0.0
        self.speed      = aggro_speed
        self.mode       = 'rôde'
        self.chase_t    = 0
        self.ambush_t   = 0
        self.ambush_x   = 0.0
        self.ambush_y   = 0.0
        self.prev_tx    = self.prev_ty = None

    def _nearest(self, fish_pos):
        if not fish_pos:
            return None, 9999.0
        dists = [(np.sqrt((self.x-fx)**2+(self.y-fy)**2), (fx,fy)) for fx,fy in fish_pos]
        dists.sort(key=lambda d: d[0])
        return dists[0][1], dists[0][0]

    def step(self, fish_pos, active, rng):
        if not active or not fish_pos:
            # Dérive lente, retour vers le centre
            cx, cy = WORLD_W/2, WORLD_H/2
            self.vx = self.vx*0.94 + (cx-self.x)/WORLD_W*0.4 + rng.uniform(-0.3,0.3)
            self.vy = self.vy*0.94 + (cy-self.y)/WORLD_H*0.4 + rng.uniform(-0.3,0.3)
            self.mode = 'rôde'
            self.chase_t = 0
            self._move()
            return

        target, d = self._nearest(fish_pos)

        if d < self.AGGRO_R:
            # ── Mode CHASSE ──────────────────────────────────────────────────
            self.mode = 'chasse'
            self.chase_t += 1
            self.ambush_t = 0

            tx, ty = target
            if self.prev_tx is not None:
                # Interception prédictive : horizon inversement proportionnel à la vitesse
                h = min(d / max(self.speed * 1.5, 0.1), 18.0)
                tx += (tx - self.prev_tx) * h
                ty += (ty - self.prev_ty) * h
                tx = float(np.clip(tx, 5, WORLD_W-5))
                ty = float(np.clip(ty, 5, WORLD_H-5))
            self.prev_tx, self.prev_ty = target

            dx = tx - self.x; dy = ty - self.y
            dn = max(np.sqrt(dx*dx+dy*dy), 1.0)
            self.vx = self.vx*0.50 + (dx/dn)*self.speed*0.50
            self.vy = self.vy*0.50 + (dy/dn)*self.speed*0.50

            # Abandon si trop loin ou trop longtemps
            if d > self.ABANDON_R or self.chase_t > self.MAX_CHASE:
                self.mode = 'embuscade'
                self.chase_t = 0
                self.prev_tx = None
                # Point d'embuscade : mi-chemin vers le centroïde des proies
                if fish_pos:
                    pcx = float(np.mean([p[0] for p in fish_pos]))
                    pcy = float(np.mean([p[1] for p in fish_pos]))
                    self.ambush_x = (self.x + pcx) / 2
                    self.ambush_y = (self.y + pcy) / 2
                else:
                    self.ambush_x = self.x
                    self.ambush_y = self.y

        elif self.mode == 'embuscade':
            # ── Mode EMBUSCADE ───────────────────────────────────────────────
            self.ambush_t += 1
            dx = self.ambush_x - self.x; dy = self.ambush_y - self.y
            dn = max(np.sqrt(dx*dx+dy*dy), 1.0)
            if dn > 8:
                # Se déplace vers le point d'embuscade lentement
                self.vx = self.vx*0.7 + (dx/dn)*self.speed*0.15
                self.vy = self.vy*0.7 + (dy/dn)*self.speed*0.15
            else:
                # Immobile, attend
                self.vx *= 0.85
                self.vy *= 0.85
            if self.ambush_t > self.AMBUSH_TICKS:
                self.mode = 'rôde'
                self.ambush_t = 0
                self.prev_tx = None

        else:
            # ── Mode RÔDE ────────────────────────────────────────────────────
            self.mode = 'rôde'
            self.prev_tx = None
            self.chase_t = 0
            # Dérive vers le centroïde des proies (vitesse 25%)
            pcx = float(np.mean([p[0] for p in fish_pos]))
            pcy = float(np.mean([p[1] for p in fish_pos]))
            dx = pcx - self.x; dy = pcy - self.y
            dn = max(np.sqrt(dx*dx+dy*dy), 1.0)
            self.vx = self.vx*0.93 + (dx/dn)*self.speed*0.25 + rng.uniform(-0.2,0.2)
            self.vy = self.vy*0.93 + (dy/dn)*self.speed*0.25 + rng.uniform(-0.2,0.2)

        self._move()

    def _move(self):
        # Clamp vitesse
        spd = np.sqrt(self.vx**2 + self.vy**2)
        if spd > self.speed * 1.5:
            self.vx *= self.speed * 1.5 / spd
            self.vy *= self.speed * 1.5 / spd
        self.x = float(np.clip(self.x + self.vx, 5, WORLD_W-5))
        self.y = float(np.clip(self.y + self.vy, 5, WORLD_H-5))


# ── Répulsion de mur (MIROIR EXACT JS) ─────────────────────────────────────────

def wall_repulse(pos, lo, hi, force=WALL_FORCE, zone=BORDER_ZONE, exp=WALL_EXP):
    dist_lo = pos - lo; dist_hi = hi - pos
    rep = 0.0
    if dist_lo < zone:
        rep += force * (1.0 - dist_lo/zone) ** exp
    if dist_hi < zone:
        rep -= force * (1.0 - dist_hi/zone) ** exp
    return rep


# ── Simulation ────────────────────────────────────────────────────────────────

def run_sim(brain, ctx, seed):
    rng = np.random.default_rng(seed)
    nf  = {"rich": 32, "normal": N_FOOD_BASE, "sparse": 10}[ctx["food"]]
    foods = [[float(rng.uniform(20, WORLD_W-20)),
              float(rng.uniform(20, WORLD_H-20)), False] for _ in range(nf)]

    pred           = Pred(ctx["pred_aggro"], rng)
    pred_active    = False
    pred_on_timer  = 0
    pred_cooldown  = int(rng.uniform(80, 220))  # premier spawn rapide

    fx = float(rng.uniform(100, WORLD_W-100))
    fy = float(rng.uniform(100, WORLD_H-100))
    vx = vy = 0.0
    fear          = 0.0
    danger_memory = 0.0   # souvenir de la dernière position du prédateur
    food_eaten    = 0
    dist_tot      = 0.0
    border_pen    = 0.0   # pénalité bord cumulée (exponentielle)
    hunger        = 1.0
    died_starvation = False
    ticks_since_meal = MEAL_SAT_TICKS  # commence "affamé depuis longtemps"
    meal_intervals = []   # historique des écarts entre repas (régularité)

    for step in range(MAX_STEPS):
        # ── Faim ──────────────────────────────────────────────────────────────
        hunger = max(0.0, hunger - HUNGER_DEC)
        if hunger <= 0.0:
            died_starvation = True
            break

        ticks_since_meal = min(MEAL_SAT_TICKS, ticks_since_meal + 1)

        # ── Timer prédateur ───────────────────────────────────────────────────
        if pred_active:
            pred_on_timer -= 1
            if pred_on_timer <= 0:
                pred_active = False
                pred_cooldown = int(rng.uniform(250, 450))
        else:
            pred_cooldown -= 1
            if pred_cooldown <= 0:
                pred_active    = True
                pred_on_timer  = int(rng.uniform(150, 250))
                side = rng.choice(["left","right","top","bottom"])
                if side == "left":
                    pred.x, pred.y = 8.0, float(rng.uniform(60, WORLD_H-60))
                elif side == "right":
                    pred.x, pred.y = WORLD_W-8.0, float(rng.uniform(60, WORLD_H-60))
                elif side == "top":
                    pred.x, pred.y = float(rng.uniform(60, WORLD_W-60)), 8.0
                else:
                    pred.x, pred.y = float(rng.uniform(60, WORLD_W-60)), WORLD_H-8.0
                pred.vx = pred.vy = 0.0
                pred.chase_t = pred.ambush_t = 0
                pred.prev_tx = pred.prev_ty = None

        # ── Inputs nourriture ─────────────────────────────────────────────────
        alive_f = [(f[0], f[1]) for f in foods if not f[2]]
        if alive_f:
            df  = [np.sqrt((fx-a[0])**2+(fy-a[1])**2) for a in alive_f]
            ci  = int(np.argmin(df))
            fd_dx = (alive_f[ci][0]-fx)/WORLD_W
            fd_dy = (alive_f[ci][1]-fy)/WORLD_H
            fd_d  = min(df[ci]/(WORLD_W*.5), 1.0)
        else:
            fd_dx = fd_dy = 0.0; fd_d = 1.0

        # ── Inputs prédateur ──────────────────────────────────────────────────
        ddx = pred.x - fx; ddy = pred.y - fy
        pd_dn = np.sqrt(ddx*ddx + ddy*ddy)
        pd_dx = ddx/WORLD_W; pd_dy = ddy/WORLD_H
        pd_d  = min(pd_dn/(WORLD_W*.5), 1.0)

        # ── Murs ──────────────────────────────────────────────────────────────
        wx = min(fx, WORLD_W-fx) / (WORLD_W*.5)
        wy = min(fy, WORLD_H-fy) / (WORLD_H*.5)

        # Pénalité bord : exponentielle avec la proximité réelle (pas binaire)
        bx = max(0.0, 1.0 - min(fx, WORLD_W-fx)/BORDER_ZONE)
        by = max(0.0, 1.0 - min(fy, WORLD_H-fy)/BORDER_ZONE)
        border_pen += max(bx, by) ** 2

        # ── Danger memory (décroît exponentiellement) ─────────────────────────
        if pred_active and pd_dn < 250:
            danger_memory = min(1.0, danger_memory + 0.20 * (1 - pd_d))
        else:
            danger_memory = max(0.0, danger_memory * 0.97)

        # ── Urgence faim ──────────────────────────────────────────────────────
        hunger_urgency = max(0.0, 1.0 - hunger * 2.5)   # monte fort sous 0.4
        meal_signal    = ticks_since_meal / MEAL_SAT_TICKS  # 0=vient de manger, 1=longtemps

        inp = np.array([
            fd_dx, fd_dy, fd_d,
            pd_dx, pd_dy, pd_d,
            wx, wy,
            vx/MAX_SPD, vy/MAX_SPD,
            fear,
            float(pred_active),
            danger_memory,
            hunger_urgency,
            meal_signal,
        ], dtype=np.float32)

        out = brain.forward(inp)
        ax  = (out[3] - out[2]) * FISH_SPD
        ay  = (out[0] - out[1]) * FISH_SPD

        # ── Répulsion mur (adoucie, mirroir exact JS) ─────────────────────────
        ax += wall_repulse(fx, 5, WORLD_W-5)
        ay += wall_repulse(fy, 5, WORLD_H-5)

        # ── Réflexe panique ───────────────────────────────────────────────────
        if pred_active and pd_dn < PANIC_DIST:
            ax = (fx-pred.x)/max(pd_dn,1)*FISH_SPD*1.6
            ay = (fy-pred.y)/max(pd_dn,1)*FISH_SPD*1.6

        vx = vx*0.50 + ax*0.50
        vy = vy*0.50 + ay*0.50
        spd = np.sqrt(vx*vx+vy*vy)
        if spd > MAX_SPD:
            vx *= MAX_SPD/spd; vy *= MAX_SPD/spd

        px, py = fx, fy
        fx = float(np.clip(fx+vx, 5, WORLD_W-5))
        fy = float(np.clip(fy+vy, 5, WORLD_H-5))
        dist_tot += np.sqrt((fx-px)**2+(fy-py)**2)

        # ── Manger ────────────────────────────────────────────────────────────
        for f in foods:
            if not f[2] and np.sqrt((fx-f[0])**2+(fy-f[1])**2) < 14:
                f[2] = True; food_eaten += 1
                hunger = min(1.0, hunger + HUNGER_GAIN)
                meal_intervals.append(ticks_since_meal)
                ticks_since_meal = 0.0
                # Respawn anti-superposition (distance min avec autres nourritures)
                for _try in range(8):
                    nx = float(rng.uniform(20, WORLD_W-20))
                    ny = float(rng.uniform(20, WORLD_H-20))
                    ok = True
                    for g in foods:
                        if g is f or g[2]:
                            continue
                        if np.sqrt((nx-g[0])**2+(ny-g[1])**2) < 28:
                            ok = False; break
                    if ok:
                        break
                f[0] = nx; f[1] = ny; f[2] = False

        # ── Prédateur step + mort ─────────────────────────────────────────────
        pred.step([(fx, fy)], pred_active, rng)
        if pred_active and np.sqrt((fx-pred.x)**2+(fy-pred.y)**2) < 18:
            sr  = step / MAX_STEPS
            fr  = food_eaten / max(step, 1) * 120
            bp  = border_pen / max(step, 1)
            fit = 0.08*sr + 0.86*min(fr,1.) - 0.06*bp
            if food_eaten == 0:
                fit *= 0.05
            return max(fit, 0.0), food_eaten, step

        # ── Peur ──────────────────────────────────────────────────────────────
        if pred_active and pd_dn < 210:
            fear = min(1.0, fear + 0.28*(1-pd_d))
        else:
            fear = max(0.0, fear - 0.04)

    # Fin de vie (survie ou famine)
    fr   = food_eaten / MAX_STEPS * 120
    eff  = food_eaten / max(dist_tot, 1) * 600
    bp   = border_pen / MAX_STEPS
    # Bonus de régularité : moyenne des écarts entre repas, normalisée
    if len(meal_intervals) >= 2:
        avg_gap = float(np.mean(meal_intervals))
        regularity = max(0.0, 1.0 - avg_gap / MEAL_SAT_TICKS)
    else:
        regularity = 0.0
    fit  = (0.70*min(fr,1.) + 0.12*min(eff,1.) + 0.08*regularity + 0.10)
    fit -= 0.16*bp            # pénalité bord (exponentielle, beaucoup plus stricte)
    if food_eaten == 0:
        fit *= 0.05
    if died_starvation:
        fit *= 0.30            # mort de faim : forte pénalité
    return max(fit, 0.0), food_eaten, MAX_STEPS


# ── Évaluation multi-seeds ────────────────────────────────────────────────────

def _eval_worker(args):
    """Worker pour multiprocessing."""
    weights_list, ctx, base, ns = args
    brain = MLP.from_list(weights_list)
    np.random.seed(base % (2**31))  # évite les collisions de seed
    fits = [run_sim(brain, ctx, base + k*1997)[0] for k in range(ns)]
    return float(np.mean(fits))

def evaluate(brain, ctx, base, ns=N_SEEDS):
    return _eval_worker((brain.to_list(), ctx, base, ns))


# ── Évolution d'une lignée ────────────────────────────────────────────────────

def evolve(name, ctx, n_gen=N_GENERATIONS, pop_size=POP_SIZE):
    print(f"\n[{name}] {ctx['desc']}")
    pop = [MLP() for _ in range(pop_size)]
    std = MUT_INIT; best_brain = pop[0]; best_fit = -1.0; hist = []; stag = 0

    for gen in range(n_gen):
        # Évaluation (parallèle si possible)
        args = [(b.to_list(), ctx, gen*pop_size+i, N_SEEDS) for i,b in enumerate(pop)]
        try:
            n_proc = min(cpu_count(), 4)
            with Pool(n_proc) as pool:
                fit_vals = pool.map(_eval_worker, args)
        except Exception:
            fit_vals = [_eval_worker(a) for a in args]

        scores = sorted(zip(fit_vals, pop), key=lambda x: x[0], reverse=True)
        gf, gb = scores[0]
        avg = float(np.mean([s[0] for s in scores]))

        if gf > best_fit + 1e-4:
            best_fit = gf; best_brain = gb; stag = 0
        else:
            stag += 1

        # Mutation adaptative
        if stag >= STAG_WINDOW:
            std = min(std*1.65, MUT_MAX); stag = 0
            print(f"    ↑ σ={std:.3f} (stagnation)")
        else:
            std = max(std*0.91, MUT_MIN)

        _, food_log, steps_log = run_sim(gb, ctx, 42)
        hist.append({"gen": gen, "best": round(gf,4), "avg": round(avg,4),
                     "food": food_log, "steps": steps_log, "sigma": round(std,4)})
        print(f"  Gen {gen+1:02d} | fit={gf:.4f} avg={avg:.4f} food={food_log} "
              f"steps={steps_log} σ={std:.3f}")

        # Sélection + reproduction
        elites = [b for _,b in scores[:ELITE_K]]
        new_pop = list(elites)
        fv = np.array([scores[i][0] for i in range(ELITE_K)])
        fv = np.maximum(fv, 1e-6)
        p  = fv / fv.sum()
        while len(new_pop) < pop_size:
            i1, i2 = np.random.choice(ELITE_K, 2, replace=False, p=p)
            new_pop.append(elites[i1].crossover(elites[i2]).mutate(std))
        pop = new_pop

    return {
        "name": name, "color": ctx["color"], "desc": ctx["desc"],
        "context": ctx, "final_fitness": round(best_fit,4),
        "weights": best_brain.to_list(), "history": hist,
        "arch": {"input": INPUT_SIZE, "hidden": HIDDEN_SIZE, "output": OUTPUT_SIZE},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42); random.seed(42)
    results = {}
    for name, ctx in LINEAGE_CONTEXTS.items():
        results[name] = evolve(name, ctx)

    out = {
        "world":    {"w": WORLD_W, "h": WORLD_H},
        "lineages": results,
        "arch":     {"input": INPUT_SIZE, "hidden": HIDDEN_SIZE, "output": OUTPUT_SIZE},
    }
    with open("lineages.json","w") as f:
        json.dump(out, f, indent=2)
    print("\n✓ lineages.json")
    for n,d in results.items():
        print(f"  {n}: {d['final_fitness']:.4f}")

if __name__ == "__main__":
    main()