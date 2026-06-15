// ══════════════════════════════════════════════════════════════
//  CONSTANTES (miroir exact du Python)
// ══════════════════════════════════════════════════════════════
const INPUT_SIZE  = 15;
const HIDDEN_SIZE = 18;
const OUTPUT_SIZE = 4;
const W = 800, H = 600;
const MAX_SPD   = 3.2;
const FISH_SPD  = 3.0;
const PANIC_DIST = 100;
const FOOD_R = 11, PRED_R = 16, FISH_R = 9;
const HUNGER_DEC  = 1.0 / (2500 * 0.75);
const HUNGER_GAIN = 0.40;
const BORDER_ZONE = 80;
const WALL_FORCE  = 2.6;   // mirroir exact Python (était 5.5 — beaucoup trop fort)
const WALL_EXP    = 1.6;   // mirroir exact Python (était 2.2)
const MEAL_SAT_TICKS = 400.0;
const FOOD_MIN_DIST = 28;  // anti-superposition nourriture
const N_FOOD_BASE = 22;
const AGGRO_R = 230, ABANDON_R = 290, MAX_CHASE_TICKS = 220;
// Nombre de nourritures par contexte de lignée (mirroir exact Python)
const FOOD_COUNT_BY_CONTEXT = { rich: 32, normal: N_FOOD_BASE, sparse: 10 };

// ══════════════════════════════════════════════════════════════
//  MLP — miroir exact
// ══════════════════════════════════════════════════════════════
function mlpForward(weights, x) {
  let idx = 0;
  const nW1 = INPUT_SIZE * HIDDEN_SIZE;
  const W1 = weights.slice(idx, idx+nW1); idx += nW1;
  const b1 = weights.slice(idx, idx+HIDDEN_SIZE); idx += HIDDEN_SIZE;
  const nW2 = HIDDEN_SIZE * OUTPUT_SIZE;
  const W2 = weights.slice(idx, idx+nW2); idx += nW2;
  const b2 = weights.slice(idx, idx+OUTPUT_SIZE);

  const h = new Float64Array(HIDDEN_SIZE);
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    let s = b1[j];
    for (let i = 0; i < INPUT_SIZE; i++) s += x[i] * W1[j*INPUT_SIZE+i];
    h[j] = Math.tanh(s);
  }
  const out = new Float64Array(OUTPUT_SIZE);
  for (let j = 0; j < OUTPUT_SIZE; j++) {
    let s = b2[j];
    for (let i = 0; i < HIDDEN_SIZE; i++) s += h[i] * W2[j*HIDDEN_SIZE+i];
    out[j] = Math.tanh(s);
  }
  return out;
}

// ══════════════════════════════════════════════════════════════
//  État global
// ══════════════════════════════════════════════════════════════
let lineageData = null;
let fishes = [], foods = [], preds = [];
let tick = 0, paused = false, speed = 1;
let hoveredFish = null;
let lastInsightUpdate = 0;

const canvas = document.getElementById('aquarium');
const ctx    = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

function resizeCanvas() {
  const wrap = canvas.parentElement;
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
function dist(x1,y1,x2,y2) { return Math.hypot(x2-x1, y2-y1); }

// ══════════════════════════════════════════════════════════════
//  Interprétation comportementale (dashboard)
// ══════════════════════════════════════════════════════════════
function behaviourInsight(fish) {
  // Calcul de métriques comportementales
  const alive         = fish.alive;
  const fe            = fish.foodEaten;
  const steps         = fish.stepsSurvived;
  const hunger        = fish.hunger;
  const fear          = fish.fearAvg;
  const deathCause    = fish.deathCause;
  const foodPerTick   = fe / Math.max(steps, 1);
  const context       = fish.context;

  if (!alive) {
    if (deathCause === 'starvation') {
      if (fe === 0)
        return "⚠ N'a pas compris que ne pas manger entraîne la mort.";
      if (fear > 0.35)
        return "La peur du prédateur a paralysé la recherche de nourriture.";
      return "A mangé, mais trop rarement — épuisement progressif.";
    }
    if (deathCause === 'predator') {
      if (fe === 0)
        return "Tué sans avoir mangé — les neurones n'ont pas encore de stratégie.";
      if (steps < 200)
        return "Mort tôt : pas eu le temps d'apprendre la fuite.";
      return `A survécu ${steps} ticks avant d'être attrapé. Progression réelle.`;
    }
  }

  // Vivant — analyse du comportement en cours
  if (steps < 100) return "Neurones en phase d'exploration initiale.";

  // Régularité alimentaire : le réseau a-t-il appris à anticiper la faim
  // plutôt que d'attendre l'urgence ?
  if (fish.mealIntervals && fish.mealIntervals.length >= 4) {
    const avgGap = fish.mealIntervals.reduce((a,b)=>a+b,0)/fish.mealIntervals.length;
    if (avgGap < 120 && fear < 0.3)
      return "✓ Rythme alimentaire régulier appris — la pression de faim ne monte presque jamais.";
    if (avgGap > 300)
      return "Repas trop espacés — la faim grimpe dangereusement entre deux prises.";
  }

  if (foodPerTick > 0.015 && fear < 0.2)
    return "✓ Excellente stratégie : mange efficacement, peu de peur inutile.";
  if (foodPerTick > 0.015 && fear > 0.4)
    return "Mange bien malgré la peur — bonne gestion du risque.";
  if (foodPerTick < 0.003 && fear > 0.4)
    return "Paralysé par la peur — survie passive, mais la faim arrive.";
  if (foodPerTick < 0.003 && hunger < 0.5)
    return "⚠ Faim critique — urgence nourriture pas encore comprise.";
  if (fe > 8 && context && context.food === 'sparse')
    return "✓ Adapté aux milieux pauvres : trouve la nourriture rare efficacement.";
  if (fe > 12)
    return "✓ Stratégie alimentaire optimisée, longues séquences de chasse.";
  if (hunger > 0.7 && fe > 3)
    return "Bon équilibre survie/nourriture — modèle viable.";

  return "Comportement stable, en cours d'optimisation.";
}

function behaviourTags(lin) {
  const c = lin.context;
  const tags = [];
  if (!c) return tags;
  if (c.pred_aggro >= 2.2) tags.push({text:'PRESSION HAUTE', color:'#e74c3c'});
  else if (c.pred_aggro <= 1.0) tags.push({text:'ENV. SAFE', color:'#2ecc71'});
  if (c.food === 'rich')   tags.push({text:'NOURRITURE ABONDANTE', color:'#f39c12'});
  if (c.food === 'sparse') tags.push({text:'NOURRITURE RARE', color:'#e67e22'});
  return tags;
}

// ══════════════════════════════════════════════════════════════
//  Spawn world — TOUS les objets correctement initialisés
// ══════════════════════════════════════════════════════════════
function makeFoodItem(pool) {
  const f = {
    x: 0, y: 0,
    eaten: false,
    phase: Math.random()*Math.PI*2,
    size:  0.8 + Math.random()*0.5,
    type:  Math.floor(Math.random()*3),  // 0=pellet 1=algue 2=plancton
  };
  placeFoodNoOverlap(f, pool || []);
  return f;
}

// Replace food coordinates with a position that respects a minimum
// distance from other (non-eaten) food in the same pool — avoids the
// "stacked algae" visual mess.
function placeFoodNoOverlap(f, pool) {
  for (let tries = 0; tries < 8; tries++) {
    const nx = 30 + Math.random()*(W-60);
    const ny = 30 + Math.random()*(H-100);
    let ok = true;
    for (let i = 0; i < pool.length; i++) {
      const g = pool[i];
      if (g === f || g.eaten) continue;
      if (dist(nx,ny,g.x,g.y) < FOOD_MIN_DIST) { ok = false; break; }
    }
    if (ok) { f.x = nx; f.y = ny; return; }
    if (tries === 7) { f.x = nx; f.y = ny; } // dernier essai : on accepte
  }
}

function spawnWorld() {
  if (!lineageData) return;
  fishes = []; foods = []; preds = []; tick = 0;

  const names = Object.keys(lineageData.lineages);
  names.forEach((name, i) => {
    const lin   = lineageData.lineages[name];
    const angle = (i/names.length)*Math.PI*2;
    const r     = Math.min(W,H)*0.28;

    // Pool de nourriture propre à cette lignée (mirroir exact entraînement)
    const foodCtx   = (lin.context && lin.context.food) || 'normal';
    const nFood     = FOOD_COUNT_BY_CONTEXT[foodCtx] !== undefined ? FOOD_COUNT_BY_CONTEXT[foodCtx] : N_FOOD_BASE;
    const foodPool  = [];
    for (let k = 0; k < nFood; k++) foodPool.push(makeFoodItem(foodPool));

    fishes.push({
      name, color:lin.color,
      // Poids : doit être un Array plat pour slicing dans mlpForward
      weights: Array.isArray(lin.weights) ? lin.weights : Array.from(lin.weights),
      x: W/2 + Math.cos(angle)*r,
      y: H/2 + Math.sin(angle)*r,
      vx:0, vy:0,
      angle: angle+Math.PI,
      tailPhase: Math.random()*Math.PI*2,
      tailAmp:   0.2,
      trail:     [],
      alive:     true,
      foodEaten: 0,
      stepsSurvived: 0,
      distanceTraveled: 0,
      hunger:    1.0,
      fear:      0.0,
      fearAccum: 0.0,
      fearAvg:   0.0,
      dangerMem: 0.0,
      timeSinceMeal: MEAL_SAT_TICKS,   // ← nouveau : pour input "meal_signal"
      mealIntervals: [],               // ← historique pour insight régularité
      deathCause: null,
      finalFitness: lin.final_fitness,
      context: lin.context,
      history: lin.history,
      insight: '',
      foodPool,                         // pool de nourriture propre à cette lignée
    });

    // Le pool global "foods" reste utilisé par le rendu (drawFood itère tous les pools)
    foods.push(...foodPool);
  });

  // Prédateur unique, mais capable de cibler n'importe quel poisson vivant
  // et d'adapter sa vitesse au contexte (pred_aggro) de SA cible actuelle —
  // mirroir fidèle de l'entraînement (chaque lignée affronte un prédateur
  // dont la vitesse correspond à pred_aggro de SON propre contexte).
  preds.push({
    x: W*0.85, y: H*0.5,
    vx:0, vy:0,
    angle: Math.PI,
    active: false,
    onTimer: 0,
    cooldown: 120 + Math.random()*100,
    mode: 'rôde',
    chaseTicks: 0,
    ambushTicks: 0,
    ambushX: W/2, ambushY: H/2,
    prevTx: null, prevTy: null,
    tailPhase: 0,
    speed: 1.8,           // valeur par défaut, réajustée dynamiquement chaque tick
    target: null,         // référence au poisson actuellement visé
  });
}

// ══════════════════════════════════════════════════════════════
//  Simulation step
// ══════════════════════════════════════════════════════════════
function simStep() {
  tick++;
  const aliveFishes = fishes.filter(f => f.alive);

  // ── Prédateur ─────────────────────────────────────────────────────────────
  preds.forEach(pred => {
    pred.tailPhase += pred.active ? 0.18 : 0.06;

    if (pred.active) {
      pred.onTimer--;
      if (pred.onTimer <= 0) {
        pred.active = false;
        pred.cooldown = 280 + Math.random()*200;
        pred.mode = 'rôde';
        pred.chaseTicks = 0;
        pred.prevTx = pred.prevTy = null;
      }
    } else {
      pred.cooldown--;
      if (pred.cooldown <= 0) {
        pred.active   = true;
        pred.onTimer  = 160 + Math.random()*100;
        pred.chaseTicks = pred.ambushTicks = 0;
        pred.prevTx = pred.prevTy = null;
        pred.mode = 'rôde';
        // Spawn depuis un bord
        const side = Math.floor(Math.random()*4);
        if      (side===0) { pred.x=8;       pred.y=60+Math.random()*(H-120); }
        else if (side===1) { pred.x=W-8;     pred.y=60+Math.random()*(H-120); }
        else if (side===2) { pred.x=60+Math.random()*(W-120); pred.y=8; }
        else               { pred.x=60+Math.random()*(W-120); pred.y=H-8; }
        pred.vx = pred.vy = 0;
      }
    }

    if (!pred.active || aliveFishes.length === 0) {
      // Dérive passive vers le centre
      pred.vx = pred.vx*0.93 + (W/2-pred.x)/W*0.5 + (Math.random()-0.5)*0.6;
      pred.vy = pred.vy*0.93 + (H/2-pred.y)/H*0.5 + (Math.random()-0.5)*0.6;
      pred.mode = 'rôde';
      pred.x = Math.max(5, Math.min(W-5, pred.x + pred.vx));
      pred.y = Math.max(5, Math.min(H-5, pred.y + pred.vy));
      if (Math.hypot(pred.vx, pred.vy) > 0.05) pred.angle = Math.atan2(pred.vy, pred.vx);
      return;
    }

    // Cible : la plus proche, avec un peu d'hystérésis pour éviter les
    // changements de cible erratiques (mirroir de la règle d'abandon
    // d'entraînement : on ne change de cible que si une proie est
    // significativement plus proche que la cible actuelle).
    let nearest = null, minD = Infinity;
    aliveFishes.forEach(f => {
      const d = dist(pred.x, pred.y, f.x, f.y);
      if (d < minD) { minD = d; nearest = f; }
    });
    if (!nearest) return;

    let target = pred.target && pred.target.alive ? pred.target : null;
    if (!target) {
      target = nearest;
      pred.prevTx = pred.prevTy = null;
    } else {
      const dCur = dist(pred.x, pred.y, target.x, target.y);
      if (nearest !== target && dist(pred.x,pred.y,nearest.x,nearest.y) < dCur*0.6) {
        target = nearest;
        pred.prevTx = pred.prevTy = null;
      } else {
        minD = dCur;
      }
    }
    pred.target = target;

    // Vitesse adaptée au pred_aggro de la lignée actuellement ciblée
    // (mirroir exact de l'entraînement : chaque lignée a appris contre
    // un prédateur à SA vitesse). Lissé pour éviter les sauts brusques.
    const targetAggro = (target.context && target.context.pred_aggro) || 1.6;
    pred.speed = pred.speed*0.9 + targetAggro*0.1;

    if (minD < AGGRO_R) {
      // ── CHASSE ──
      pred.mode = 'chasse';
      pred.chaseTicks++;
      pred.ambushTicks = 0;

      let tx = target.x, ty = target.y;
      if (pred.prevTx !== null) {
        const h = Math.min(minD / Math.max(pred.speed*1.5, 0.1), 18);
        tx += (tx - pred.prevTx)*h;
        ty += (ty - pred.prevTy)*h;
        tx = Math.max(5, Math.min(W-5, tx));
        ty = Math.max(5, Math.min(H-5, ty));
      }
      pred.prevTx = target.x;
      pred.prevTy = target.y;

      const dx = tx-pred.x, dy = ty-pred.y;
      const dn = Math.max(Math.hypot(dx,dy), 1);
      pred.vx = pred.vx*0.50 + (dx/dn)*pred.speed*0.50;
      pred.vy = pred.vy*0.50 + (dy/dn)*pred.speed*0.50;

      if (minD > ABANDON_R || pred.chaseTicks > MAX_CHASE_TICKS) {
        pred.mode = 'embuscade';
        pred.chaseTicks = 0;
        pred.prevTx = pred.prevTy = null;
        // Point d'embuscade : mi-chemin vers le centroïde
        const pcx = aliveFishes.reduce((s,f)=>s+f.x, 0)/aliveFishes.length;
        const pcy = aliveFishes.reduce((s,f)=>s+f.y, 0)/aliveFishes.length;
        pred.ambushX = (pred.x + pcx)/2;
        pred.ambushY = (pred.y + pcy)/2;
        pred.ambushTicks = 0;
      }
    } else if (pred.mode === 'embuscade') {
      // ── EMBUSCADE ──
      pred.ambushTicks++;
      const dx = pred.ambushX - pred.x, dy = pred.ambushY - pred.y;
      const dn = Math.max(Math.hypot(dx,dy), 1);
      if (dn > 10) {
        pred.vx = pred.vx*0.7 + (dx/dn)*pred.speed*0.12;
        pred.vy = pred.vy*0.7 + (dy/dn)*pred.speed*0.12;
      } else {
        pred.vx *= 0.8; pred.vy *= 0.8;
      }
      if (pred.ambushTicks > 90 + Math.random()*60) {
        pred.mode = 'rôde';
        pred.ambushTicks = 0;
        pred.prevTx = pred.prevTy = null;
      }
    } else {
      // ── RÔDE ──
      pred.mode = 'rôde';
      pred.chaseTicks = 0; pred.prevTx = pred.prevTy = null;
      const pcx = aliveFishes.reduce((s,f)=>s+f.x,0)/aliveFishes.length;
      const pcy = aliveFishes.reduce((s,f)=>s+f.y,0)/aliveFishes.length;
      const dx = pcx-pred.x, dy = pcy-pred.y;
      const dn = Math.max(Math.hypot(dx,dy),1);
      pred.vx = pred.vx*0.92 + (dx/dn)*pred.speed*0.22 + (Math.random()-0.5)*0.3;
      pred.vy = pred.vy*0.92 + (dy/dn)*pred.speed*0.22 + (Math.random()-0.5)*0.3;
    }

    // Clamp vitesse prédateur
    const pspd = Math.hypot(pred.vx, pred.vy);
    if (pspd > pred.speed*1.6) { pred.vx *= pred.speed*1.6/pspd; pred.vy *= pred.speed*1.6/pspd; }
    pred.x = Math.max(5, Math.min(W-5, pred.x + pred.vx));
    pred.y = Math.max(5, Math.min(H-5, pred.y + pred.vy));
    if (pspd > 0.05) pred.angle = Math.atan2(pred.vy, pred.vx);
  });

  // ── Poissons ──────────────────────────────────────────────────────────────
  fishes.forEach(fish => {
    if (!fish.alive) return;
    fish.stepsSurvived++;
    fish.tailPhase += 0.16;

    // Faim
    fish.hunger = Math.max(0, fish.hunger - HUNGER_DEC);
    if (fish.hunger <= 0) {
      fish.alive = false;
      fish.deathCause = 'starvation';
      return;
    }

    const aliveFoods = fish.foodPool.filter(f => !f.eaten);

    // Inputs nourriture
    let fdDx=0, fdDy=0, fdDist=1;
    if (aliveFoods.length > 0) {
      let closest=null, minD2=Infinity;
      aliveFoods.forEach(f => {
        const d2 = dist(fish.x,fish.y,f.x,f.y);
        if (d2 < minD2) { minD2=d2; closest=f; }
      });
      fdDx   = (closest.x-fish.x)/W;
      fdDy   = (closest.y-fish.y)/H;
      fdDist = Math.min(minD2/(W*0.5), 1);
    }

    // Inputs prédateur
    let pdDx=0, pdDy=0, pdDist=1, minPD=Infinity, closestPred=null;
    preds.forEach(p => {
      const d = dist(fish.x,fish.y,p.x,p.y);
      if (d < minPD) { minPD=d; closestPred=p; }
    });
    if (closestPred) {
      pdDx   = (closestPred.x-fish.x)/W;
      pdDy   = (closestPred.y-fish.y)/H;
      pdDist = Math.min(minPD/(W*0.5), 1);
    }

    // Murs
    const wx = Math.min(fish.x, W-fish.x)/(W*0.5);
    const wy = Math.min(fish.y, H-fish.y)/(H*0.5);

    // Danger memory
    const isPredActive = closestPred && closestPred.active;
    if (isPredActive && minPD < 260) {
      fish.dangerMem = Math.min(1, fish.dangerMem + 0.18*(1-pdDist));
    } else {
      fish.dangerMem = Math.max(0, fish.dangerMem*0.97);
    }

    // Urgence faim
    const hungerUrgency = Math.max(0, 1.0 - fish.hunger*2.5);

    // Temps écoulé depuis le dernier repas (sature à 1.0) — permet au
    // réseau d'apprendre une dynamique anticipative de la faim plutôt
    // que de réagir seulement quand hungerUrgency explose.
    fish.timeSinceMeal = Math.min(MEAL_SAT_TICKS, fish.timeSinceMeal + 1);
    const mealSignal = fish.timeSinceMeal / MEAL_SAT_TICKS;

    const inputs = [
      fdDx, fdDy, fdDist,
      pdDx, pdDy, pdDist,
      wx, wy,
      fish.vx/MAX_SPD, fish.vy/MAX_SPD,
      fish.fear,
      isPredActive ? 1.0 : 0.0,
      fish.dangerMem,
      hungerUrgency,
      mealSignal,
    ];

    const out = mlpForward(fish.weights, inputs);
    let ax = (out[3]-out[2])*FISH_SPD;
    let ay = (out[0]-out[1])*FISH_SPD;

    // Répulsion mur — adoucie, mirroir exact du Python (était force=5.5,
    // exp=2.2, ce qui pouvait écraser totalement la décision du réseau
    // dans les coins et bloquer le poisson en oscillation contre le bord).
    function wallRep(pos, lo, hi, zone=BORDER_ZONE, force=WALL_FORCE, exp=WALL_EXP) {
      let r=0;
      const dlo = pos-lo, dhi = hi-pos;
      if (dlo < zone) r += force*(1-dlo/zone)**exp;
      if (dhi < zone) r -= force*(1-dhi/zone)**exp;
      return r;
    }
    ax += wallRep(fish.x, 5, W-5);
    ay += wallRep(fish.y, 5, H-5);

    // Réflexe panique
    if (isPredActive && minPD < PANIC_DIST) {
      const dxp=fish.x-closestPred.x, dyp=fish.y-closestPred.y;
      const d = Math.max(Math.hypot(dxp,dyp),1);
      ax = (dxp/d)*FISH_SPD*1.6;
      ay = (dyp/d)*FISH_SPD*1.6;
    }

    fish.vx = fish.vx*0.50 + ax*0.50;
    fish.vy = fish.vy*0.50 + ay*0.50;
    const spd = Math.hypot(fish.vx, fish.vy);
    if (spd > MAX_SPD) { fish.vx *= MAX_SPD/spd; fish.vy *= MAX_SPD/spd; }

    const prevX=fish.x, prevY=fish.y;
    fish.x = Math.max(5, Math.min(W-5, fish.x+fish.vx));
    fish.y = Math.max(5, Math.min(H-5, fish.y+fish.vy));
    const moved = Math.hypot(fish.x-prevX, fish.y-prevY);
    fish.distanceTraveled += moved;
    fish.tailAmp = Math.min(1, fish.tailAmp*0.88 + moved*0.18);
    if (Math.abs(fish.vx)>0.01||Math.abs(fish.vy)>0.01)
      fish.angle = Math.atan2(fish.vy, fish.vx);

    // Trail
    fish.trail.push({x:fish.x, y:fish.y});
    if (fish.trail.length > 20) fish.trail.shift();

    // Manger (recyclage sur place, anti-superposition dans le pool de la lignée)
    aliveFoods.forEach(f => {
      if (!f.eaten && dist(fish.x,fish.y,f.x,f.y) < FISH_R+FOOD_R) {
        fish.foodEaten++;
        fish.hunger = Math.min(1.0, fish.hunger+HUNGER_GAIN);
        fish.mealIntervals.push(fish.timeSinceMeal);
        if (fish.mealIntervals.length > 30) fish.mealIntervals.shift();
        fish.timeSinceMeal = 0;
        // Recycle la même entrée, position anti-superposition
        f.phase = Math.random()*Math.PI*2;
        f.type  = Math.floor(Math.random()*3);
        placeFoodNoOverlap(f, fish.foodPool);
        // f.eaten reste false : pas de pool infini
      }
    });

    // Mort par prédateur
    preds.forEach(pred => {
      if (pred.active && dist(fish.x,fish.y,pred.x,pred.y) < FISH_R+PRED_R) {
        fish.alive = false;
        fish.deathCause = 'predator';
      }
    });

    // Peur
    if (isPredActive && minPD < 200) {
      fish.fear = Math.min(1, fish.fear + 0.28*(1-pdDist));
    } else {
      fish.fear = Math.max(0, fish.fear - 0.04);
    }
    fish.fearAccum += fish.fear;
    fish.fearAvg = fish.fearAccum / Math.max(fish.stepsSurvived,1);
  });
}

// ══════════════════════════════════════════════════════════════
//  Décor statique
// ══════════════════════════════════════════════════════════════
const CORALS=[], ROCKS=[], SAND_RIPPLES=[];

function generateDecor() {
  CORALS.length=0; ROCKS.length=0; SAND_RIPPLES.length=0;
  const CC=['#ff6b6b','#ff9f43','#ffd32a','#c8d6e5','#a29bfe','#fd79a8','#00b894'];
  for (let i=0;i<10;i++) CORALS.push({
    x:Math.random()*(W-80)+40, h:22+Math.random()*48,
    branches:2+Math.floor(Math.random()*4),
    color:CC[Math.floor(Math.random()*CC.length)],
    type:Math.floor(Math.random()*3), phase:Math.random()*Math.PI*2
  });
  for (let i=0;i<7;i++) {
    const pts=[], n=6+Math.floor(Math.random()*4), r=16+Math.random()*30;
    for (let j=0;j<n;j++) {
      const a=(j/n)*Math.PI*2, rr=r*(0.65+Math.random()*0.65);
      pts.push([Math.cos(a)*rr, Math.sin(a)*rr*0.5]);
    }
    ROCKS.push({x:Math.random()*(W-60)+30, pts, dark:Math.random()>0.5});
  }
  for (let i=0;i<18;i++) SAND_RIPPLES.push({
    x:Math.random()*W, w:25+Math.random()*55, phase:Math.random()*Math.PI*2
  });
}

// ══════════════════════════════════════════════════════════════
//  Bulles
// ══════════════════════════════════════════════════════════════
const bubbles = Array.from({length:40}, ()=>({
  x:Math.random()*W, y:Math.random()*H,
  r:Math.random()*2+0.35,
  vy:-(Math.random()*0.38+0.07),
  vx:(Math.random()-0.5)*0.1,
  opacity:Math.random()*0.12+0.03,
  wobble:Math.random()*Math.PI*2,
}));

function drawBubbles() {
  bubbles.forEach(b => {
    b.wobble+=0.038; b.y+=b.vy; b.x+=b.vx+Math.sin(b.wobble)*0.13;
    if (b.y<-5){b.y=H+5;b.x=Math.random()*W;}
    const sx=(b.x/W)*canvas.width, sy=(b.y/H)*canvas.height;
    ctx.beginPath(); ctx.arc(sx,sy,b.r,0,Math.PI*2);
    ctx.strokeStyle=`rgba(120,210,255,${b.opacity})`; ctx.lineWidth=0.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(sx-b.r*.3,sy-b.r*.3,b.r*.3,0,Math.PI*2);
    ctx.fillStyle=`rgba(200,240,255,${b.opacity*1.4})`; ctx.fill();
  });
}

// ══════════════════════════════════════════════════════════════
//  Background
// ══════════════════════════════════════════════════════════════
function drawBackground() {
  const cw=canvas.width, ch=canvas.height;
  const g=ctx.createLinearGradient(0,0,0,ch);
  g.addColorStop(0,'#010612'); g.addColorStop(.4,'#020c1e');
  g.addColorStop(.82,'#03101f'); g.addColorStop(1,'#010810');
  ctx.fillStyle=g; ctx.fillRect(0,0,cw,ch);

  // Rayons
  ctx.save();
  for(let i=0;i<6;i++){
    const bx=(0.1+i*0.16)*cw, sw=Math.sin(tick*.0024+i*1.3)*.04*cw, x=bx+sw;
    const bw=(11+Math.sin(tick*.002+i*.7)*3)*(cw/800);
    const gd=ctx.createLinearGradient(x,0,x,ch*.7);
    gd.addColorStop(0,`rgba(28,115,195,${0.038+Math.sin(tick*.003+i)*.01})`);
    gd.addColorStop(.5,'rgba(8,55,130,.013)'); gd.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=gd;
    ctx.beginPath(); ctx.moveTo(x-bw,0); ctx.lineTo(x+bw,0);
    ctx.lineTo(x+bw*.35+sw*.25,ch*.7); ctx.lineTo(x-bw*.35+sw*.25,ch*.7); ctx.fill();
  }
  ctx.restore();

  // Caustics
  ctx.save(); ctx.globalAlpha=0.022;
  for(let i=0;i<10;i++){
    const t=tick*.002+i*.55;
    const x=(0.5+.5*Math.sin(t*1.1))*cw, y=(0.15+.1*Math.sin(t*.9+1))*ch;
    const r=(18+10*Math.sin(t*.7))*(cw/800);
    const gc=ctx.createRadialGradient(x,y,0,x,y,r);
    gc.addColorStop(0,'#3ab4ff'); gc.addColorStop(1,'rgba(58,180,255,0)');
    ctx.fillStyle=gc; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();

  // Sable
  const sandY=ch*.87;
  const sg=ctx.createLinearGradient(0,sandY,0,ch);
  sg.addColorStop(0,'#0c1c0f'); sg.addColorStop(.3,'#101808'); sg.addColorStop(1,'#09140e');
  ctx.fillStyle=sg; ctx.beginPath(); ctx.moveTo(0,sandY);
  for(let x=0;x<=cw;x+=10) ctx.lineTo(x,sandY+Math.sin(x*.017+tick*.001)*2.5);
  ctx.lineTo(cw,ch); ctx.lineTo(0,ch); ctx.closePath(); ctx.fill();

  ctx.save(); ctx.globalAlpha=0.055;
  SAND_RIPPLES.forEach(r=>{
    const sx=(r.x/W)*cw, sy=sandY+7+Math.sin(r.phase+tick*.0007)*2.5;
    const sw=(r.w/W)*cw;
    ctx.beginPath(); ctx.ellipse(sx,sy,sw,2.2,0,0,Math.PI*2);
    ctx.strokeStyle='#3a5e40'; ctx.lineWidth=1; ctx.stroke();
  });
  ctx.restore();
}

// ══════════════════════════════════════════════════════════════
//  Décor — coraux et rochers
// ══════════════════════════════════════════════════════════════
function drawDecor() {
  const cw=canvas.width, ch=canvas.height;
  const sx=cw/W, sy=ch/H, sandY=ch*.87;

  ROCKS.forEach(rock=>{
    const rx=rock.x*sx, ry=sandY+8;
    ctx.save(); ctx.translate(rx,ry);
    ctx.beginPath();
    rock.pts.forEach(([px,py],i)=>i===0?ctx.moveTo(px*sx,py*sy):ctx.lineTo(px*sx,py*sy));
    ctx.closePath();
    ctx.fillStyle=rock.dark?'#0c1812':'#111e15';
    ctx.strokeStyle='#192d1c'; ctx.lineWidth=1; ctx.fill(); ctx.stroke();
    ctx.restore();
  });

  CORALS.forEach(coral=>{
    const cx2=coral.x*sx, sway=Math.sin(tick*.008+coral.phase)*2.5;
    ctx.save(); ctx.translate(cx2,sandY+2);
    ctx.shadowBlur=9; ctx.shadowColor=coral.color+'44';
    if(coral.type===0) {
      drawCoralBranch(ctx,0,0,-Math.PI/2,coral.h*sy*.75,coral.branches,coral.color,sway,0);
    } else if(coral.type===1) {
      const fw=coral.h*sx*.45, fh=coral.h*sy*.65;
      for(let i=0;i<=8;i++){
        const t2=i/8, x2=(t2-.5)*fw, sw2=sway*(1-t2*.3);
        ctx.beginPath(); ctx.moveTo(sw2*.2,0);
        ctx.quadraticCurveTo(x2*.4+sw2*.6,-fh*.5,x2+sw2,-fh);
        ctx.strokeStyle=coral.color+'99'; ctx.lineWidth=1+(1-Math.abs(t2-.5)*2)*1.4; ctx.stroke();
      }
    } else {
      for(let b=0;b<coral.branches+1;b++){
        const bx=(b-coral.branches/2)*9*sx, bh=(coral.h*.5+b*7)*sy*.65, bs=sway*(.5+b*.08);
        ctx.beginPath(); ctx.moveTo(bx,0); ctx.lineTo(bx+bs,-bh);
        ctx.strokeStyle=coral.color+'bb'; ctx.lineWidth=3.5*sx; ctx.lineCap='round'; ctx.stroke();
        ctx.beginPath(); ctx.arc(bx+bs,-bh,3.5*sx,0,Math.PI*2);
        ctx.fillStyle=coral.color; ctx.fill();
      }
    }
    ctx.restore();
  });
}
function drawCoralBranch(c,x,y,angle,length,depth,color,sway,lvl){
  if(depth<=0||length<2.5)return;
  const sf=sway*(lvl*.28+0.18);
  const ex=x+Math.cos(angle+sf*.04)*length, ey=y+Math.sin(angle+sf*.04)*length;
  c.beginPath(); c.moveTo(x,y);
  c.quadraticCurveTo(x+Math.cos(angle)*length*.5+sf*.45,y+Math.sin(angle)*length*.5,ex,ey);
  c.strokeStyle=color+(lvl===0?'bb':'88'); c.lineWidth=Math.max(.7,(depth-lvl)*1.1);
  c.lineCap='round'; c.stroke();
  if(depth===1){c.beginPath();c.arc(ex,ey,2.2,0,Math.PI*2);c.fillStyle=color;c.fill();}
  drawCoralBranch(c,ex,ey,angle-.44,length*.64,depth-1,color,sway,lvl+1);
  drawCoralBranch(c,ex,ey,angle+.44,length*.64,depth-1,color,sway,lvl+1);
  if(depth>1&&Math.random()<.45) drawCoralBranch(c,ex,ey,angle,length*.52,depth-1,color,sway,lvl+1);
}

// ══════════════════════════════════════════════════════════════
//  Nourriture
// ══════════════════════════════════════════════════════════════
function drawFood() {
  foods.forEach(f => {
    if (f.eaten) return;
    f.phase = (f.phase||0) + 0.05;
    const sx=(f.x/W)*canvas.width, sy=(f.y/H)*canvas.height;
    const s=f.size||1;
    ctx.save(); ctx.translate(sx,sy);

    if (f.type===0) {
      const p=0.87+Math.sin(f.phase)*.13;
      ctx.beginPath(); ctx.arc(0,0,4.5*s*p,0,Math.PI*2);
      ctx.fillStyle='#39ff8a'; ctx.shadowBlur=9; ctx.shadowColor='#39ff8a'; ctx.fill();
      ctx.globalAlpha=.35; ctx.beginPath(); ctx.arc(-1,-1.2,1.4*s,0,Math.PI*2);
      ctx.fillStyle='#afffcc'; ctx.fill();
    } else if (f.type===1) {
      ctx.strokeStyle='#2ecc71'; ctx.lineWidth=1.5;
      ctx.shadowBlur=5; ctx.shadowColor='#2ecc71'; ctx.globalAlpha=.82;
      const h2=11*s;
      ctx.beginPath(); ctx.moveTo(0,h2/2);
      for(let i=0;i<=5;i++){
        const t2=i/5, wx2=Math.sin(t2*Math.PI*2+f.phase)*2.8*s;
        ctx.lineTo(wx2, h2/2-t2*h2);
      }
      ctx.stroke();
    } else {
      ctx.shadowBlur=7; ctx.shadowColor='#00ffd4';
      for(let i=0;i<4;i++){
        const a=(i/4)*Math.PI*2+f.phase*.45, r2=3.8*s;
        const p2=.8+Math.sin(f.phase+i*1.1)*.2;
        ctx.beginPath(); ctx.arc(Math.cos(a)*r2,Math.sin(a)*r2,1.7*p2,0,Math.PI*2);
        ctx.fillStyle='#00ffd4'; ctx.fill();
      }
    }
    ctx.restore();
  });
}

// ══════════════════════════════════════════════════════════════
//  Prédateur requin — avec mode visible
// ══════════════════════════════════════════════════════════════
function drawPredators() {
  const scX=canvas.width/W, scY=canvas.height/H;
  preds.forEach(pred => {
    const sx=pred.x*scX, sy=pred.y*scY;
    const isActive=pred.active;
    const isChasing=pred.mode==='chasse';
    const isAmbush=pred.mode==='embuscade';
    const tailSway=Math.sin(pred.tailPhase)*(isChasing?.42:isAmbush?.06:.15);

    ctx.save(); ctx.translate(sx,sy); ctx.rotate(pred.angle||0);
    ctx.globalAlpha = isActive ? (isChasing?.95:.70) : .35;
    const L = isChasing?34:isActive?28:22;

    if (isChasing){ctx.shadowBlur=24;ctx.shadowColor='rgba(200,0,0,.55)';}
    else if(isAmbush){ctx.shadowBlur=10;ctx.shadowColor='rgba(140,0,0,.3)';}

    // Queue
    ctx.save(); ctx.rotate(tailSway);
    ctx.beginPath();
    ctx.moveTo(-L*.55,0); ctx.lineTo(-L,-L*.36); ctx.lineTo(-L*.95,0); ctx.lineTo(-L,L*.36);
    ctx.closePath();
    ctx.fillStyle=isChasing?'#991111':isActive?'#5a1515':'#2d0d0d'; ctx.fill();
    ctx.restore();

    // Corps
    ctx.beginPath();
    ctx.moveTo(L,0);
    ctx.bezierCurveTo(L*.6,-L*.22,-L*.2,-L*.25,-L*.55,-L*.04);
    ctx.bezierCurveTo(-L*.6,-L*.02,-L*.6,L*.02,-L*.55,L*.04);
    ctx.bezierCurveTo(-L*.2,L*.25,L*.6,L*.22,L,0);
    ctx.fillStyle=isChasing?'#b21515':isActive?'#4a1212':'#2a0d0d'; ctx.fill();

    // Ventre
    ctx.beginPath();
    ctx.moveTo(L*.7,0);ctx.bezierCurveTo(L*.4,-L*.1,-L*.1,-L*.1,-L*.28,-L*.02);
    ctx.bezierCurveTo(-L*.28,L*.02,-L*.1,L*.1,L*.4,L*.1); ctx.closePath();
    ctx.fillStyle=isChasing?'#cc2a2a44':'#3d181844'; ctx.fill();

    // Nageoire dorsale
    ctx.beginPath();
    ctx.moveTo(L*.06,-L*.2);ctx.lineTo(L*.26,-L*.5);ctx.lineTo(L*.4,-L*.22);
    ctx.closePath();
    ctx.fillStyle=isChasing?'#991111':isActive?'#3d1818':'#2a0d0d'; ctx.fill();

    // Nageoire pectorale
    ctx.beginPath();
    ctx.moveTo(L*.2,L*.15);ctx.lineTo(L*.05,L*.42);ctx.lineTo(-L*.12,L*.18);
    ctx.closePath();
    ctx.fillStyle=isActive?'#aa1818':'#300d0d'; ctx.fill();

    // Oeil
    ctx.shadowBlur=0;
    ctx.beginPath();ctx.arc(L*.55,-L*.07,L*.07,0,Math.PI*2);
    ctx.fillStyle='#080808'; ctx.fill();
    ctx.beginPath();ctx.arc(L*.55,-L*.07,L*.03,0,Math.PI*2);
    ctx.fillStyle=isChasing?'#ff1010':isAmbush?'#ff8800':'#333'; ctx.fill();

    // Dents si en chasse
    if(isChasing){
      ctx.globalAlpha=0.65;
      for(let t=0;t<3;t++){
        const tx2=L*.87-t*L*.09, tw=L*.04;
        ctx.beginPath();ctx.moveTo(tx2,-L*.03);ctx.lineTo(tx2-tw,-L*.1);ctx.lineTo(tx2-tw*2,-L*.03);
        ctx.fillStyle='#f0f0f0'; ctx.fill();
      }
    }

    // Icône mode (embuscade = ! )
    if(isAmbush){
      ctx.globalAlpha=0.9; ctx.shadowBlur=0;
      ctx.fillStyle='#ff8800'; ctx.font=`bold ${Math.round(L*.7)}px monospace`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText('!', 0, -L*1.2);
    }

    ctx.restore();
  });
}

// ══════════════════════════════════════════════════════════════
//  Poisson
// ══════════════════════════════════════════════════════════════
function drawFish(fish, scX, scY) {
  const sx=fish.x*scX, sy=fish.y*scY;

  // Trail
  if(fish.alive && fish.trail && fish.trail.length>3){
    ctx.save();
    for(let i=1;i<fish.trail.length;i++){
      const t=i/fish.trail.length;
      ctx.beginPath();
      ctx.moveTo(fish.trail[i-1].x*scX, fish.trail[i-1].y*scY);
      ctx.lineTo(fish.trail[i].x*scX,   fish.trail[i].y*scY);
      ctx.strokeStyle=fish.color;
      ctx.globalAlpha=t*0.065*(fish.fear>0.35?1.6:1);
      ctx.lineWidth=t*2.2; ctx.stroke();
    }
    ctx.restore();
  }

  ctx.save(); ctx.translate(sx,sy); ctx.rotate(fish.angle||0);
  const alive=fish.alive;
  ctx.globalAlpha=alive?1:.16;
  const L=11;
  const tailSway=alive?Math.sin(fish.tailPhase||0)*(fish.tailAmp||0)*0.75:0;

  // Couleur modifiée par la faim
  let bodyColor=fish.color;
  if(alive && fish.hunger < 0.35){
    // Teinte vers le gris/jaune quand affamé
    const t=1-(fish.hunger/0.35);
    bodyColor=lerpColor(fish.color,'#888855',t*0.55);
  }

  const fearGlow=fish.fear>0.4;
  if(alive){
    ctx.shadowBlur=fearGlow?16:8;
    ctx.shadowColor=fearGlow?'#ff4444':bodyColor;
  }

  // Queue
  ctx.save(); ctx.translate(-L*1.1,0); ctx.rotate(tailSway*.7);
  ctx.beginPath();
  ctx.moveTo(0,0);ctx.lineTo(-L*.72,-L*.68);ctx.lineTo(-L*.38,0);ctx.lineTo(-L*.72,L*.68);
  ctx.closePath(); ctx.fillStyle=bodyColor+'bb'; ctx.fill();
  ctx.restore();

  // Corps
  ctx.beginPath();
  ctx.moveTo(L*1.2,0);
  ctx.bezierCurveTo(L*.8,-L*.6,-L*.5,-L*.68,-L*1.1,0);
  ctx.bezierCurveTo(-L*.5,L*.68,L*.8,L*.6,L*1.2,0);
  ctx.fillStyle=bodyColor; ctx.fill();

  // Reflet ventre
  ctx.beginPath();
  ctx.moveTo(L*.9,0);ctx.bezierCurveTo(L*.5,-L*.27,-L*.2,-L*.28,-L*.68,0);
  ctx.bezierCurveTo(-L*.2,L*.28,L*.5,L*.27,L*.9,0);
  ctx.fillStyle='rgba(255,255,255,0.11)'; ctx.fill();

  // Nageoire dorsale
  ctx.beginPath();
  ctx.moveTo(-L*.28,-L*.48);ctx.quadraticCurveTo(L*.1,-L*.98,L*.48,-L*.5);
  ctx.lineTo(L*.48,-L*.37);ctx.quadraticCurveTo(L*.1,-L*.68,-L*.28,-L*.37);
  ctx.closePath(); ctx.globalAlpha=alive?.68:.14;
  ctx.fillStyle=bodyColor+'aa'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  // Nageoire anale
  ctx.beginPath();
  ctx.moveTo(-L*.08,L*.40);ctx.quadraticCurveTo(L*.14,L*.82,L*.44,L*.43);
  ctx.lineTo(L*.44,L*.33);ctx.quadraticCurveTo(L*.14,L*.58,-L*.08,L*.30);
  ctx.closePath(); ctx.globalAlpha=alive?.52:.1;
  ctx.fillStyle=bodyColor+'88'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  // Nageoire pectorale
  ctx.shadowBlur=0;
  ctx.beginPath();ctx.moveTo(L*.38,L*.17);ctx.quadraticCurveTo(L*.68,L*.52,L*.08,L*.48);
  ctx.closePath(); ctx.globalAlpha=alive?.48:.08;
  ctx.fillStyle=bodyColor+'77'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  // Œil
  ctx.shadowBlur=0;
  ctx.beginPath();ctx.arc(L*.54,-L*.14,L*.21,0,Math.PI*2);
  ctx.fillStyle='#0c190c'; ctx.fill();
  ctx.beginPath();ctx.arc(L*.54,-L*.14,L*.13,0,Math.PI*2);
  ctx.fillStyle=alive?'#ffffff':'#333'; ctx.globalAlpha=alive?.88:.28; ctx.fill();
  ctx.globalAlpha=alive?1:.16;
  ctx.beginPath();ctx.arc(L*.58,-L*.17,L*.055,0,Math.PI*2);
  ctx.fillStyle='#090909'; ctx.fill();

  // Écailles
  if(alive){
    ctx.globalAlpha=0.07; ctx.strokeStyle='#fff'; ctx.lineWidth=0.55;
    for(let i=0;i<3;i++){
      ctx.beginPath();ctx.arc(-L*.18+i*L*.44,0,L*.28-i*.01,-Math.PI*.5,Math.PI*.5);ctx.stroke();
    }
    ctx.globalAlpha=1;
  }

  // Barre de peur (rouge)
  if(alive && fish.fear>0.18){
    ctx.shadowBlur=0;
    const bw=L*2.3;
    ctx.globalAlpha=fish.fear*.7;
    ctx.fillStyle='#ff2222';
    ctx.fillRect(-bw/2,-L*1.45,bw*fish.fear,2.2);
    ctx.strokeStyle='rgba(255,255,255,.12)';ctx.lineWidth=.4;
    ctx.strokeRect(-bw/2,-L*1.45,bw,2.2);
    ctx.globalAlpha=alive?1:.16;
  }

  // Barre de faim (jaune→rouge)
  if(alive){
    ctx.shadowBlur=0;
    const bw=L*2.3;
    const h=fish.hunger;
    const hColor = h>0.5 ? '#f0c030' : h>0.25 ? '#e08020' : '#cc2020';
    ctx.globalAlpha=Math.max(0, 0.85 - h*0.5);  // plus visible quand faim haute
    ctx.fillStyle=hColor;
    ctx.fillRect(-bw/2,-L*1.75,bw*h,2);
    ctx.strokeStyle='rgba(255,255,255,.08)';ctx.lineWidth=.4;
    ctx.strokeRect(-bw/2,-L*1.75,bw,2);
    ctx.globalAlpha=1;
  }

  ctx.restore();

  // Label hover
  if(hoveredFish===fish){
    ctx.save(); ctx.globalAlpha=.9;
    ctx.fillStyle=fish.color; ctx.font='10px JetBrains Mono,monospace';
    ctx.fillText(fish.name,sx+15,sy-15); ctx.restore();
  }
}

function lerpColor(c1, c2, t) {
  const r1=parseInt(c1.slice(1,3),16),g1=parseInt(c1.slice(3,5),16),b1=parseInt(c1.slice(5,7),16);
  const r2=parseInt(c2.slice(1,3),16),g2=parseInt(c2.slice(3,5),16),b2=parseInt(c2.slice(5,7),16);
  const r=Math.round(r1+(r2-r1)*t),g=Math.round(g1+(g2-g1)*t),b=Math.round(b1+(b2-b1)*t);
  return '#'+r.toString(16).padStart(2,'0')+g.toString(16).padStart(2,'0')+b.toString(16).padStart(2,'0');
}

// ══════════════════════════════════════════════════════════════
//  Dessin de la scène
// ══════════════════════════════════════════════════════════════
function drawScene() {
  resizeCanvas();
  const scX=canvas.width/W, scY=canvas.height/H;
  drawBackground();
  drawDecor();
  ctx.save(); drawBubbles(); ctx.restore();
  drawFood();
  drawPredators();
  fishes.forEach(f => drawFish(f, scX, scY));
}

// ══════════════════════════════════════════════════════════════
//  UI / Dashboard
// ══════════════════════════════════════════════════════════════
function buildLineageCards() {
  if (!lineageData) return;
  const container = document.getElementById('lineageCards');
  container.innerHTML = '';
  Object.values(lineageData.lineages).forEach(lin => {
    const card = document.createElement('div');
    card.className = 'lineage-card';
    card.id = 'card-'+lin.name;
    card.style.setProperty('--lcolor', lin.color);
    const tags = behaviourTags(lin);
    const tagHTML = tags.map(t=>
      `<span class="behav-tag" style="color:${t.color};border-color:${t.color}33">${t.text}</span>`
    ).join(' ');
    card.innerHTML = `
      <div class="lineage-header">
        <span class="lineage-name">${lin.name}</span>
        <span class="lineage-status" id="status-${lin.name}">vivant</span>
      </div>
      <div class="stat-row"><span>fitness évol.</span><span class="stat-val">${lin.final_fitness.toFixed(3)}</span></div>
      <div class="stat-row"><span>nourriture</span><span class="stat-val" id="food-${lin.name}">0</span></div>
      <div class="stat-row"><span>survie</span><span class="stat-val" id="steps-${lin.name}">0 t</span></div>
      <div class="stat-row"><span>peur moy.</span><span class="stat-val" id="fear-${lin.name}">0%</span></div>
      <div class="hunger-bar-wrap">
        <span style="font-size:9px;color:var(--muted)">faim</span>
        <div class="hunger-bar-bg">
          <div class="hunger-bar-fill" id="hunger-${lin.name}" style="width:100%;background:#f0c030"></div>
        </div>
        <span class="hunger-label" id="hunger-pct-${lin.name}">100%</span>
      </div>
      ${tagHTML}
      <div class="behav-insight" id="insight-${lin.name}">Initialisation…</div>
      <canvas class="sparkline" id="spark-${lin.name}" height="26"></canvas>
    `;
    container.appendChild(card);
    requestAnimationFrame(() => drawSparkline(lin));
  });
}

function drawSparkline(lin) {
  const cvs = document.getElementById('spark-'+lin.name);
  if (!cvs) return;
  cvs.width = cvs.offsetWidth; cvs.height = 26;
  const c = cvs.getContext('2d');
  const data = lin.history.map(h=>h.best);
  if (!data.length) return;
  const mx = Math.max(...data, 0.01);
  const cw=cvs.width, ch=cvs.height;
  c.clearRect(0,0,cw,ch);
  const grad=c.createLinearGradient(0,0,0,ch);
  grad.addColorStop(0,lin.color+'2a'); grad.addColorStop(1,'transparent');
  c.fillStyle=grad; c.beginPath();
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*cw, y=ch-(v/mx)*ch*.86-2;
    i===0?c.moveTo(x,ch):undefined;
    i===0?c.lineTo(x,y):c.lineTo(x,y);
  });
  c.lineTo(cw,ch); c.closePath(); c.fill();
  c.strokeStyle=lin.color; c.lineWidth=1.4;
  c.shadowBlur=5; c.shadowColor=lin.color; c.globalAlpha=.82;
  c.beginPath();
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*cw, y=ch-(v/mx)*ch*.86-2;
    i===0?c.moveTo(x,y):c.lineTo(x,y);
  });
  c.stroke();
}

function updateUI(force=false) {
  const alive = fishes.filter(f=>f.alive).length;
  document.getElementById('gTick').textContent = tick;
  document.getElementById('gAlive').textContent = `${alive}/${fishes.length}`;
  document.getElementById('gFood').textContent  = foods.filter(f=>!f.eaten).length;
  const pred = preds[0];
  if (pred) {
    const modeLabels = {'chasse':'⚡ chasse','embuscade':'👁 embuscade','rôde':'~ rôde'};
    document.getElementById('gPred').textContent =
      pred.active ? (modeLabels[pred.mode]||pred.mode) : '· inactif';
  }

  fishes.forEach(fish => {
    const sEl  = document.getElementById('status-'+fish.name);
    const fEl  = document.getElementById('food-'+fish.name);
    const stEl = document.getElementById('steps-'+fish.name);
    const feEl = document.getElementById('fear-'+fish.name);
    const hBar = document.getElementById('hunger-'+fish.name);
    const hPct = document.getElementById('hunger-pct-'+fish.name);
    const ins  = document.getElementById('insight-'+fish.name);

    if (sEl) {
      const deathLabel = fish.deathCause === 'starvation' ? '✕ famine' : '✕ prédateur';
      sEl.textContent  = fish.alive ? '● vivant' : deathLabel;
      sEl.style.color  = fish.alive ? '#2ecc71' : (fish.deathCause==='starvation'?'#e67e22':'#e74c3c');
    }
    if (fEl)  fEl.textContent  = fish.foodEaten;
    if (stEl) stEl.textContent = fish.stepsSurvived+'t';
    if (feEl) feEl.textContent = Math.round(fish.fearAvg*100)+'%';

    if (hBar && hPct) {
      const hp = Math.round(fish.hunger*100);
      hBar.style.width = hp+'%';
      hBar.style.background = fish.hunger > 0.5 ? '#f0c030' : fish.hunger > 0.25 ? '#e08020' : '#cc2020';
      hPct.textContent = hp+'%';
    }

    // Insight : mise à jour toutes les 2s ou si mort récente
    if (ins && (force || !fish.alive || tick - lastInsightUpdate > 120)) {
      ins.textContent = behaviourInsight(fish);
    }
  });
  if (force || tick - lastInsightUpdate > 120) lastInsightUpdate = tick;
}

// ══════════════════════════════════════════════════════════════
//  Mouse
// ══════════════════════════════════════════════════════════════
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX-rect.left)/canvas.width*W;
  const my = (e.clientY-rect.top)/canvas.height*H;
  hoveredFish = null;
  fishes.forEach(f => { if(f.alive && dist(mx,my,f.x,f.y)<22) hoveredFish=f; });
  if (hoveredFish) {
    tooltip.style.display='block';
    tooltip.style.left=(e.clientX-rect.left+16)+'px';
    tooltip.style.top=(e.clientY-rect.top+16)+'px';
    const f=hoveredFish;
    const fd={rich:'abondante',normal:'normale',sparse:'rare'};
    const c=f.context;
    tooltip.innerHTML=`
      <b style="color:${f.color}">${f.name}</b><br>
      nourriture : <b>${f.foodEaten}</b> (${fd[c&&c.food]||'?'})<br>
      faim : <b>${Math.round(f.hunger*100)}%</b><br>
      depuis dernier repas : ${Math.round(f.timeSinceMeal)}t<br>
      peur actuelle : <b>${Math.round(f.fear*100)}%</b><br>
      peur moyenne : ${Math.round(f.fearAvg*100)}%<br>
      danger memory : ${Math.round(f.dangerMem*100)}%<br>
      vitesse prédateur (cible) : ${preds[0]&&preds[0].target===f ? preds[0].speed.toFixed(2) : '—'}<br>
      fitness évol : ${f.finalFitness.toFixed(3)}
    `;
  } else {
    tooltip.style.display='none';
  }
});
canvas.addEventListener('mouseleave', ()=>{tooltip.style.display='none';hoveredFish=null;});

// ══════════════════════════════════════════════════════════════
//  Contrôles
// ══════════════════════════════════════════════════════════════
function setSpeed(s) {
  speed=s;
  ['btn1x','btn3x','btn10x'].forEach(id=>document.getElementById(id).classList.remove('active'));
  document.getElementById('btn'+s+'x').classList.add('active');
}
function togglePause() {
  paused=!paused;
  document.getElementById('btnPause').textContent=paused?'▶ play':'⏸ pause';
}
function resetSim() { generateDecor(); spawnWorld(); updateUI(true); }

// ══════════════════════════════════════════════════════════════
//  Boucle
// ══════════════════════════════════════════════════════════════
let lastUITs=0;
function loop(ts) {
  if(!paused) for(let i=0;i<speed;i++) simStep();
  drawScene();
  if(ts-lastUITs>110){updateUI();lastUITs=ts;}
  requestAnimationFrame(loop);
}

async function init() {
  generateDecor();
  try {
    const res = await fetch('lineages.json');
    lineageData = await res.json();
    buildLineageCards();
    spawnWorld();
    updateUI(true);
    requestAnimationFrame(loop);
  } catch(e) {
    document.body.innerHTML+=`
      <div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
        background:#020810;border:1px solid #e74c3c;padding:24px 32px;
        color:#e74c3c;font-family:monospace;border-radius:4px;text-align:center">
        ⚠ lineages.json introuvable<br><br>
        <span style="color:#888">Lance d'abord :</span><br>
        <code style="color:#aaa">python evolution.py</code>
      </div>`;
  }
}
init();