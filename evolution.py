import numpy as np
import json
import random

WORLD_W, WORLD_H = 800, 600
N_FOOD_BASE      = 22
MAX_STEPS        = 2500
N_GENERATIONS    = 20
POP_SIZE         = 20
ELITE_K          = 4
N_SEEDS          = 3
MUT_INIT         = 0.13
MUT_MIN          = 0.03
MUT_MAX          = 0.30
STAG_WINDOW      = 4

# ── Architecture (miroir exact dans le HTML) ──────────────────────────────────
# 12 inputs :
#  0-2  : nourriture (dx/W, dy/H, dist_norm)
#  3-5  : prédateur  (dx/W, dy/H, dist_norm)
#  6-7  : murs proches (dist_x_norm, dist_y_norm)
#  8-9  : vélocité propre (vx/MAX_SPD, vy/MAX_SPD)
#  10   : peur [0,1]
#  11   : pred_active flag [0,1]
INPUT_SIZE  = 12
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4    # [avant, arrière, gauche, droite]
MAX_SPD     = 3.2
FISH_SPD    = 3.0
PANIC_DIST  = 100

LINEAGE_CONTEXTS = {
    "Rouge":  {"pred_aggro": 2.6, "food": "normal", "color": "#e74c3c",
               "desc": "Prédateur agressif dès gen 0 — sélection fuite"},
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
        n = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + OUTPUT_SIZE
        if w is None:
            s1 = np.sqrt(2/(INPUT_SIZE+HIDDEN_SIZE))
            s2 = np.sqrt(2/(HIDDEN_SIZE+OUTPUT_SIZE))
            self.w = np.concatenate([
                np.random.randn(INPUT_SIZE*HIDDEN_SIZE)*s1,
                np.zeros(HIDDEN_SIZE),
                np.random.randn(HIDDEN_SIZE*OUTPUT_SIZE)*s2,
                np.zeros(OUTPUT_SIZE)
            ])
        else:
            self.w = w.copy()

    def forward(self, x):
        i = 0
        W1 = self.w[i:i+INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE,HIDDEN_SIZE); i+=INPUT_SIZE*HIDDEN_SIZE
        b1 = self.w[i:i+HIDDEN_SIZE]; i+=HIDDEN_SIZE
        W2 = self.w[i:i+HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE,OUTPUT_SIZE); i+=HIDDEN_SIZE*OUTPUT_SIZE
        b2 = self.w[i:i+OUTPUT_SIZE]
        h  = np.tanh(x@W1+b1)
        return np.tanh(h@W2+b2)

    def mutate(self, std):
        return MLP(self.w + np.random.randn(*self.w.shape)*std)

    def crossover(self, other):
        a = np.random.uniform(0.3,0.7)
        return MLP(a*self.w + (1-a)*other.w)

    def to_list(self): return self.w.tolist()

# ── Prédateur prédictif ───────────────────────────────────────────────────────
class Pred:
    AGGRO_R   = 220
    ABANDON_R = 270
    MAX_CHASE = 200

    def __init__(self, aggro, rng):
        self.x   = float(rng.uniform(50, WORLD_W-50))
        self.y   = float(rng.uniform(50, WORLD_H-50))
        self.vx  = self.vy = 0.0
        self.spd = aggro
        self.mode= "rôde"
        self.chase_t = 0
        self.prev_tx = self.prev_ty = None

    def step(self, fish_pos, active, rng):
        if not active or not fish_pos:
            # dérive lente
            self.vx = self.vx*.94 + rng.uniform(-.3,.3)
            self.vy = self.vy*.94 + rng.uniform(-.3,.3)
            self.mode = "rôde"
        else:
            dists = [np.sqrt((self.x-fx)**2+(self.y-fy)**2) for fx,fy in fish_pos]
            ci    = int(np.argmin(dists)); tx,ty = fish_pos[ci]; d = dists[ci]
            if d < self.AGGRO_R:
                self.mode = "chasse"; self.chase_t += 1
                # interception prédictive
                if self.prev_tx is not None:
                    h = min(d/max(self.spd,0.1), 15)
                    tx += (tx-self.prev_tx)*h
                    ty += (ty-self.prev_ty)*h
                self.prev_tx, self.prev_ty = fish_pos[ci]
                dx=tx-self.x; dy=ty-self.y; dn=max(np.sqrt(dx*dx+dy*dy),1)
                self.vx = self.vx*.55+(dx/dn)*self.spd*.45
                self.vy = self.vy*.55+(dy/dn)*self.spd*.45
                if d>self.ABANDON_R or self.chase_t>self.MAX_CHASE:
                    self.mode="rôde"; self.chase_t=0
            else:
                self.mode="rôde"; self.chase_t=0; self.prev_tx=None
                cx=np.mean([p[0] for p in fish_pos]); cy=np.mean([p[1] for p in fish_pos])
                dx=cx-self.x; dy=cy-self.y; dn=max(np.sqrt(dx*dx+dy*dy),1)
                self.vx=self.vx*.92+(dx/dn)*.5; self.vy=self.vy*.92+(dy/dn)*.5
        self.x=float(np.clip(self.x+self.vx,5,WORLD_W-5))
        self.y=float(np.clip(self.y+self.vy,5,WORLD_H-5))

# ── Simulation ────────────────────────────────────────────────────────────────
def run_sim(brain, ctx, seed):
    rng = np.random.default_rng(seed)
    nf  = {"rich":30,"normal":N_FOOD_BASE,"sparse":10}[ctx["food"]]
    foods= [[float(rng.uniform(20,WORLD_W-20)), float(rng.uniform(20,WORLD_H-20)), False]
            for _ in range(nf)]

    pred = Pred(ctx["pred_aggro"], rng)
    # prédateur actif par fenêtres : spawn ~toutes les 400 ticks, reste 120-200 ticks
    pred_active   = False
    pred_on_timer = 0
    pred_cooldown = int(rng.uniform(100, 300))   # démarrage décalé

    fx=float(rng.uniform(80,WORLD_W-80)); fy=float(rng.uniform(80,WORLD_H-80))
    vx=vy=0.0; fear=0.0; food_eaten=0; dist_tot=0.0

    for step in range(MAX_STEPS):
        # timer prédateur
        if pred_active:
            pred_on_timer -= 1
            if pred_on_timer <= 0:
                pred_active = False
                pred_cooldown = int(rng.uniform(300, 500))
        else:
            pred_cooldown -= 1
            if pred_cooldown <= 0:
                pred_active   = True
                pred_on_timer = int(rng.uniform(120, 200))
                pred.x = float(rng.choice([-1,1])) * rng.uniform(WORLD_W*.5, WORLD_W*.9)
                pred.x = float(np.clip(pred.x, 10, WORLD_W-10))
                pred.y = float(rng.uniform(50, WORLD_H-50))

        # nourriture la plus proche
        alive_f = [(f[0],f[1]) for f in foods if not f[2]]
        if alive_f:
            df = [np.sqrt((fx-a[0])**2+(fy-a[1])**2) for a in alive_f]
            ci = int(np.argmin(df))
            fd_dx=(alive_f[ci][0]-fx)/WORLD_W; fd_dy=(alive_f[ci][1]-fy)/WORLD_H
            fd_d=min(df[ci]/(WORLD_W*.5),1.)
        else:
            fd_dx=fd_dy=0.; fd_d=1.

        # prédateur
        ddx=pred.x-fx; ddy=pred.y-fy
        pd_dn=np.sqrt(ddx*ddx+ddy*ddy)
        pd_dx=ddx/WORLD_W; pd_dy=ddy/WORLD_H; pd_d=min(pd_dn/(WORLD_W*.5),1.)

        wx=min(fx,WORLD_W-fx)/(WORLD_W*.5); wy=min(fy,WORLD_H-fy)/(WORLD_H*.5)

        inp=np.array([fd_dx,fd_dy,fd_d,
                      pd_dx,pd_dy,pd_d,
                      wx,wy,
                      vx/MAX_SPD, vy/MAX_SPD,
                      fear,float(pred_active)],dtype=np.float32)

        out=brain.forward(inp)
        ax=(out[3]-out[2])*FISH_SPD; ay=(out[0]-out[1])*FISH_SPD

        # réflexe panique
        if pred_active and pd_dn < PANIC_DIST:
            ax=(fx-pred.x)/max(pd_dn,1)*FISH_SPD*1.5
            ay=(fy-pred.y)/max(pd_dn,1)*FISH_SPD*1.5

        vx=vx*.5+ax*.5; vy=vy*.5+ay*.5
        spd=np.sqrt(vx*vx+vy*vy)
        if spd>MAX_SPD: vx*=MAX_SPD/spd; vy*=MAX_SPD/spd

        px,py=fx,fy
        fx=float(np.clip(fx+vx,5,WORLD_W-5)); fy=float(np.clip(fy+vy,5,WORLD_H-5))
        dist_tot+=np.sqrt((fx-px)**2+(fy-py)**2)

        # manger
        for f in foods:
            if not f[2] and np.sqrt((fx-f[0])**2+(fy-f[1])**2)<14:
                f[2]=True; food_eaten+=1
                foods.append([float(rng.uniform(20,WORLD_W-20)),float(rng.uniform(20,WORLD_H-20)),False])

        # prédateur step + mort
        pred.step([(fx,fy)], pred_active, rng)
        if pred_active and np.sqrt((fx-pred.x)**2+(fy-pred.y)**2)<17:
            sr=step/MAX_STEPS
            fr=food_eaten/max(step,1)*100   # nourriture par 100 ticks
            fit=.08*sr + .92*min(fr,1.)
            return fit,food_eaten,step

        # peur
        if pred_active and pd_dn<200:
            fear=min(1.,fear+.30*(1-pd_d))
        else:
            fear=max(0.,fear-.05)

    fr=food_eaten/MAX_STEPS*100
    eff=food_eaten/max(dist_tot,1)*500
    fit=.80*min(fr,1.) + .10*min(eff,1.) + .10
    return fit,food_eaten,MAX_STEPS

def evaluate(brain, ctx, base, ns=N_SEEDS):
    return float(np.mean([run_sim(brain,ctx,base+k*997)[0] for k in range(ns)]))

def evolve(name, ctx, n_gen=N_GENERATIONS, pop_size=POP_SIZE):
    print(f"\n[{name}] {ctx['desc']}")
    pop=[MLP() for _ in range(pop_size)]
    std=MUT_INIT; best_brain=pop[0]; best_fit=-1.; hist=[]; stag=0

    for gen in range(n_gen):
        scores=[(evaluate(b,ctx,gen*pop_size+i),b) for i,b in enumerate(pop)]
        scores.sort(key=lambda x:x[0],reverse=True)
        gf,gb=scores[0]; avg=float(np.mean([s[0] for s in scores]))

        if gf>best_fit+1e-4: best_fit=gf; best_brain=gb; stag=0
        else: stag+=1
        if stag>=STAG_WINDOW: std=min(std*1.6,MUT_MAX); stag=0; print(f"    ↑ σ={std:.3f}")
        else: std=max(std*.90,MUT_MIN)

        _,food_log,steps_log=run_sim(gb,ctx,42)
        hist.append({"gen":gen,"best":round(gf,4),"avg":round(avg,4),"food":food_log,"steps":steps_log,"sigma":round(std,4)})
        print(f"  Gen {gen+1:02d} | fit={gf:.4f} avg={avg:.4f} food={food_log} steps={steps_log} σ={std:.3f}")

        elites=[b for _,b in scores[:ELITE_K]]
        new_pop=list(elites)
        fv=np.array([scores[i][0] for i in range(ELITE_K)]); p=fv/fv.sum()
        while len(new_pop)<pop_size:
            i1,i2=np.random.choice(ELITE_K,2,replace=False,p=p)
            new_pop.append(elites[i1].crossover(elites[i2]).mutate(std))
        pop=new_pop

    return {"name":name,"color":ctx["color"],"desc":ctx["desc"],"context":ctx,
            "final_fitness":round(best_fit,4),"weights":best_brain.to_list(),"history":hist,
            "arch":{"input":INPUT_SIZE,"hidden":HIDDEN_SIZE,"output":OUTPUT_SIZE}}

def main():
    np.random.seed(42); random.seed(42)
    results={}
    for name,ctx in LINEAGE_CONTEXTS.items():
        results[name]=evolve(name,ctx)

    out={"world":{"w":WORLD_W,"h":WORLD_H},"lineages":results,
         "arch":{"input":INPUT_SIZE,"hidden":HIDDEN_SIZE,"output":OUTPUT_SIZE}}
    with open("lineages.json","w") as f:
        json.dump(out,f,indent=2)
    print("\n✓ lineages.json")
    for n,d in results.items():
        print(f"  {n}: {d['final_fitness']:.4f}")

if __name__=="__main__":
    main()