// ══════════════════════════════════════════════════════════════
//  AQUARIUM ÉVOLUTIF v6 — Neuroévolution en direct
// ══════════════════════════════════════════════════════════════
//
// Changement de paradigme vs v5 :
//  - Plus de "5 lignées" pré-entraînées hors-ligne. Chaque poisson
//    a son PROPRE génome (poids MLP + vecteur de tempérament),
//    hérité et muté à la reproduction. Tout évolue EN DIRECT,
//    dans le navigateur, à partir de poids aléatoires.
//  - Reproduction sexuée : deux poissons matures, proches, et bien
//    nourris produisent un enfant (crossover + mutation des deux
//    génomes parents).
//  - Régulation écologique douce : le taux de respawn de nourriture
//    diminue quand la population est dense → pression de sélection
//    naturelle sans plafond dur.
//  - Input social (16e) : direction/densité du groupe de voisins,
//    pondérée par le trait de tempérament "socialPull".
//
// evolution.py n'est plus un prérequis : c'est un outil d'analyse
// headless optionnel (statistiques de convergence sur plusieurs runs).
//
// ══════════════════════════════════════════════════════════════
//  CONSTANTES DU MONDE
// ══════════════════════════════════════════════════════════════
const INPUT_SIZE  = 16;
const HIDDEN_SIZE = 18;
const OUTPUT_SIZE = 4;
const W = 800, H = 600;
const MAX_SPD   = 3.2;
const FISH_SPD  = 3.0;
const PANIC_DIST_BASE = 100;
const FOOD_R = 11, PRED_R = 16, FISH_R = 9;
const BORDER_ZONE = 80;
const WALL_FORCE  = 2.6;
const WALL_EXP    = 1.6;
const MEAL_SAT_TICKS = 400.0;
const FOOD_MIN_DIST = 28;

// ── Écologie / population ───────────────────────────────────────
const N_FOOD        = 34;             // pool de nourriture global, fixe
const FOOD_RESPAWN_BASE = 0.045;      // proba de respawn d'une unité mangée, par tick, à population nulle
const ECO_K         = 9;              // densité de population "de référence" pour la régularisation
const HUNGER_GAIN   = 0.40;

// ── Cycle de vie ─────────────────────────────────────────────────
const START_POP      = 9;
const MIN_POP        = 3;     // si on tombe sous ce seuil, on réintroduit un individu (évite extinction totale)
const MATURITY_AGE   = 600;   // ticks avant de pouvoir se reproduire
const REPRO_ENERGY   = 0.72;  // énergie minimale pour se reproduire
const REPRO_COST     = 0.30;  // énergie perdue par parent à la reproduction
const REPRO_COOLDOWN = 500;   // ticks avant de pouvoir se reproduire à nouveau
const REPRO_RANGE    = 42;    // distance max entre 2 parents
const MAX_AGE        = 9000;  // au-delà, probabilité de mort naturelle croissante
const OLD_AGE_DEATH_RATE = 0.0006; // proba/tick de mort naturelle au-delà de MAX_AGE

// ── Génétique ─────────────────────────────────────────────────────
const MUT_STD       = 0.12;
const MUT_BIG_PROB  = 0.05;
const MUT_BIG_MULT  = 3.0;
const TEMPERAMENT_MUT_STD = 0.06;

// ══════════════════════════════════════════════════════════════
//  MLP
// ══════════════════════════════════════════════════════════════
const N_W1 = INPUT_SIZE * HIDDEN_SIZE;
const N_B1 = HIDDEN_SIZE;
const N_W2 = HIDDEN_SIZE * OUTPUT_SIZE;
const N_B2 = OUTPUT_SIZE;
const N_WEIGHTS = N_W1 + N_B1 + N_W2 + N_B2;

function randomWeights() {
  const s1 = Math.sqrt(2.0/(INPUT_SIZE+HIDDEN_SIZE));
  const s2 = Math.sqrt(2.0/(HIDDEN_SIZE+OUTPUT_SIZE));
  const w = new Float64Array(N_WEIGHTS);
  let idx=0;
  for (let i=0;i<N_W1;i++) w[idx++] = gaussian()*s1;
  for (let i=0;i<N_B1;i++) w[idx++] = 0;
  for (let i=0;i<N_W2;i++) w[idx++] = gaussian()*s2;
  for (let i=0;i<N_B2;i++) w[idx++] = 0;
  return w;
}

function gaussian() {
  // Box-Muller
  let u=0,v=0;
  while(u===0) u=Math.random();
  while(v===0) v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

function mlpForward(weights, x) {
  let idx = 0;
  const W1 = weights.subarray(idx, idx+N_W1); idx += N_W1;
  const b1 = weights.subarray(idx, idx+N_B1); idx += N_B1;
  const W2 = weights.subarray(idx, idx+N_W2); idx += N_W2;
  const b2 = weights.subarray(idx, idx+N_B2);

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

// Crossover BLX-alpha entre deux vecteurs de poids
function crossoverWeights(wa, wb) {
  const a = 0.25 + Math.random()*0.5; // a in [0.25, 0.75]
  const out = new Float64Array(N_WEIGHTS);
  for (let i=0;i<N_WEIGHTS;i++) out[i] = a*wa[i] + (1-a)*wb[i];
  return out;
}

function mutateWeights(w, std) {
  const out = new Float64Array(N_WEIGHTS);
  for (let i=0;i<N_WEIGHTS;i++) {
    let n = gaussian()*std;
    if (Math.random() < MUT_BIG_PROB) n *= MUT_BIG_MULT;
    out[i] = w[i] + n;
  }
  return out;
}

// ══════════════════════════════════════════════════════════════
//  TEMPÉRAMENT — vecteur hérité de traits comportementaux
// ══════════════════════════════════════════════════════════════
// Chaque trait dans [0,1]. Modulent à la fois les INPUTS perçus par
// le réseau (perception, social) et la PHYSIQUE du poisson
// (métabolisme, hardiesse) — donc même un réseau identique peut
// produire un comportement différent selon le tempérament porté.
//
//  perception : 0 = myope (bruit de capteur fort sur nourriture/prédateur)
//               1 = clairvoyant (perception quasi exacte)
//  metabolism : 0 = lent/économe (faim baisse lentement, vitesse -10%)
//               1 = rapide/glouton (faim baisse vite, vitesse +10%,
//                   mais gain de nourriture légèrement supérieur)
//  socialPull : 0 = solitaire (ignore le groupe)
//               1 = grégaire (fortement attiré par le centroïde des voisins)
//  boldness   : 0 = anxieux (peur monte plus vite, panique de plus loin)
//               1 = téméraire (peur monte lentement, panique seulement
//                   au dernier moment — plus risqué près du prédateur)

function randomTemperament() {
  return {
    perception: Math.random(),
    metabolism: Math.random(),
    socialPull: Math.random(),
    boldness:   Math.random(),
  };
}

function crossoverTemperament(ta, tb) {
  const out = {};
  for (const k of Object.keys(ta)) {
    const a = Math.random();
    out[k] = clamp01(a*ta[k] + (1-a)*tb[k]);
  }
  return out;
}

function mutateTemperament(t) {
  const out = {};
  for (const k of Object.keys(t)) {
    out[k] = clamp01(t[k] + gaussian()*TEMPERAMENT_MUT_STD);
  }
  return out;
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

// Couleur dérivée du tempérament — pour identifier visuellement les
// "espèces"/profils émergents sans avoir de lignées figées.
// Teinte = perception (bleu froid = myope, jaune chaud = clairvoyant)
// Saturation = boldness, Luminosité = socialPull (offset léger)
function temperamentColor(t) {
  const hue = 200 - t.perception*180;       // 200° (bleu) → 20° (orange)
  const sat = 45 + t.boldness*45;           // 45–90%
  const light = 48 + (t.socialPull-0.5)*14; // 41–55%
  return `hsl(${hue.toFixed(0)} ${sat.toFixed(0)}% ${light.toFixed(0)}%)`;
}

function temperamentLabel(t) {
  // Étiquette descriptive courte du profil dominant
  const tags = [];
  tags.push(t.perception < 0.35 ? 'myope' : t.perception > 0.65 ? 'clairvoyant' : null);
  tags.push(t.metabolism < 0.35 ? 'économe' : t.metabolism > 0.65 ? 'glouton' : null);
  tags.push(t.socialPull < 0.35 ? 'solitaire' : t.socialPull > 0.65 ? 'grégaire' : null);
  tags.push(t.boldness   < 0.35 ? 'anxieux'   : t.boldness   > 0.65 ? 'téméraire' : null);
  const present = tags.filter(Boolean);
  return present.length ? present.join(' · ') : 'équilibré';
}

// ══════════════════════════════════════════════════════════════
//  État global
// ══════════════════════════════════════════════════════════════
let fishes = [], foods = [], preds = [];
let tick = 0, paused = false, speed = 1;
let hoveredFish = null;
let nextFishId = 1;
let genCounter = 0;       // génération max atteinte (profondeur d'arbre généalogique)
let stats = {
  births: 0, deaths: { starvation:0, predator:0, old_age:0 },
  popHistory: [], // {tick, pop, avgEnergy, avgGen}
};
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
//  Nourriture — anti-superposition
// ══════════════════════════════════════════════════════════════
function makeFoodItem(pool) {
  const f = {
    x:0, y:0, eaten:false,
    phase: Math.random()*Math.PI*2,
    size:  0.8 + Math.random()*0.5,
    type:  Math.floor(Math.random()*3),
  };
  placeFoodNoOverlap(f, pool || []);
  return f;
}
function placeFoodNoOverlap(f, pool) {
  for (let tries=0; tries<8; tries++) {
    const nx = 30 + Math.random()*(W-60);
    const ny = 30 + Math.random()*(H-100);
    let ok = true;
    for (let i=0;i<pool.length;i++) {
      const g = pool[i];
      if (g===f || g.eaten) continue;
      if (dist(nx,ny,g.x,g.y) < FOOD_MIN_DIST) { ok=false; break; }
    }
    if (ok || tries===7) { f.x=nx; f.y=ny; return; }
  }
}

// ══════════════════════════════════════════════════════════════
//  Genèse d'un poisson
// ══════════════════════════════════════════════════════════════
function makeFish(opts) {
  const t = opts.temperament || randomTemperament();
  return {
    id: nextFishId++,
    gen: opts.gen || 0,
    weights: opts.weights || randomWeights(),
    temperament: t,
    color: temperamentColor(t),
    x: opts.x !== undefined ? opts.x : 60+Math.random()*(W-120),
    y: opts.y !== undefined ? opts.y : 60+Math.random()*(H-120),
    vx:0, vy:0,
    angle: Math.random()*Math.PI*2,
    tailPhase: Math.random()*Math.PI*2,
    tailAmp: 0.2,
    trail: [],
    alive: true,
    age: 0,
    energy: opts.energy !== undefined ? opts.energy : 1.0,
    foodEaten: 0,
    stepsSurvived: 0,
    distanceTraveled: 0,
    fear: 0, fearAccum: 0, fearAvg: 0,
    dangerMem: 0,
    timeSinceMeal: MEAL_SAT_TICKS*0.4,
    mealIntervals: [],
    reproCooldown: opts.reproCooldown || MATURITY_AGE*0.3,
    children: 0,
    deathCause: null,
    parents: opts.parents || null,
  };
}

// ══════════════════════════════════════════════════════════════
//  Spawn world
// ══════════════════════════════════════════════════════════════
function spawnWorld() {
  fishes = []; foods = []; preds = []; tick = 0; genCounter = 0;
  nextFishId = 1;
  stats = { births:0, deaths:{starvation:0,predator:0,old_age:0}, popHistory:[] };

  for (let i=0;i<START_POP;i++) fishes.push(makeFish({}));
  for (let i=0;i<N_FOOD;i++) foods.push(makeFoodItem(foods));

  preds.push({
    x: W*0.85, y: H*0.5, vx:0, vy:0, angle: Math.PI,
    active:false, onTimer:0, cooldown: 120+Math.random()*100,
    mode:'rôde', chaseTicks:0, ambushTicks:0,
    ambushX:W/2, ambushY:H/2, prevTx:null, prevTy:null,
    tailPhase:0, speed:1.6, target:null,
  });
}

// ══════════════════════════════════════════════════════════════
//  Simulation step
// ══════════════════════════════════════════════════════════════
function simStep() {
  tick++;
  const aliveFishes = fishes.filter(f => f.alive);
  const pop = aliveFishes.length;

  // ── Régulation écologique : respawn de nourriture modulé par densité ──
  const respawnP = FOOD_RESPAWN_BASE * (ECO_K / (ECO_K + pop));
  foods.forEach(f => {
    if (f.eaten && Math.random() < respawnP) {
      placeFoodNoOverlap(f, foods);
      f.eaten = false;
      f.phase = Math.random()*Math.PI*2;
      f.type = Math.floor(Math.random()*3);
    }
  });

  // ── Prédateur ──────────────────────────────────────────────────────────
  preds.forEach(pred => stepPredator(pred, aliveFishes));

  // ── Poissons : décisions, physique, faim, peur ───────────────────────────
  fishes.forEach(fish => {
    if (!fish.alive) return;
    stepFish(fish, aliveFishes);
  });

  // ── Reproduction (après que tout le monde ait bougé) ─────────────────────
  handleReproduction();

  // ── Mortalité naturelle (vieillesse) ──────────────────────────────────────
  fishes.forEach(fish => {
    if (!fish.alive) return;
    if (fish.age > MAX_AGE) {
      const p = OLD_AGE_DEATH_RATE * (1 + (fish.age-MAX_AGE)/MAX_AGE);
      if (Math.random() < p) {
        fish.alive = false;
        fish.deathCause = 'old_age';
        stats.deaths.old_age++;
      }
    }
  });

  // ── Anti-extinction : si la population tombe sous MIN_POP, réintroduire ──
  const stillAlive = fishes.filter(f=>f.alive).length;
  if (stillAlive < MIN_POP) {
    fishes.push(makeFish({ gen: genCounter }));
    stats.births++;
  }

  // ── Nettoyage périodique des cadavres trop vieux dans l'historique ───────
  if (fishes.length > 60) {
    // Garde tous les vivants + les 20 morts les plus récentes (pour l'UI)
    const alive = fishes.filter(f=>f.alive);
    const dead  = fishes.filter(f=>!f.alive).slice(-20);
    fishes = alive.concat(dead);
  }

  // ── Historique de population (pour sparkline globale) ─────────────────────
  if (tick % 20 === 0) {
    const a = fishes.filter(f=>f.alive);
    const avgEnergy = a.length ? a.reduce((s,f)=>s+f.energy,0)/a.length : 0;
    const avgGen    = a.length ? a.reduce((s,f)=>s+f.gen,0)/a.length : 0;
    stats.popHistory.push({tick, pop:a.length, avgEnergy, avgGen});
    if (stats.popHistory.length > 250) stats.popHistory.shift();
  }
}

// ══════════════════════════════════════════════════════════════
//  Prédateur — mêmes 3 modes que v5, vitesse globale fixe + légère
//  adaptation à l'audace moyenne de la population (plus la population
//  est "téméraire" en moyenne, plus le prédateur est rapide — pression
//  de sélection auto-ajustée).
// ══════════════════════════════════════════════════════════════
const AGGRO_R = 230, ABANDON_R = 290, MAX_CHASE_TICKS = 220;
const PRED_BASE_SPEED = 1.6;

function stepPredator(pred, aliveFishes) {
  pred.tailPhase += pred.active ? 0.18 : 0.06;

  if (pred.active) {
    pred.onTimer--;
    if (pred.onTimer <= 0) {
      pred.active = false;
      pred.cooldown = 280 + Math.random()*200;
      pred.mode = 'rôde';
      pred.chaseTicks = 0;
      pred.prevTx = pred.prevTy = null;
      pred.target = null;
    }
  } else {
    pred.cooldown--;
    if (pred.cooldown <= 0) {
      pred.active = true;
      pred.onTimer = 160 + Math.random()*100;
      pred.chaseTicks = pred.ambushTicks = 0;
      pred.prevTx = pred.prevTy = null;
      pred.mode = 'rôde';
      const side = Math.floor(Math.random()*4);
      if      (side===0) { pred.x=8;       pred.y=60+Math.random()*(H-120); }
      else if (side===1) { pred.x=W-8;     pred.y=60+Math.random()*(H-120); }
      else if (side===2) { pred.x=60+Math.random()*(W-120); pred.y=8; }
      else                { pred.x=60+Math.random()*(W-120); pred.y=H-8; }
      pred.vx = pred.vy = 0;
    }
  }

  if (!pred.active || aliveFishes.length === 0) {
    pred.vx = pred.vx*0.93 + (W/2-pred.x)/W*0.5 + (Math.random()-0.5)*0.6;
    pred.vy = pred.vy*0.93 + (H/2-pred.y)/H*0.5 + (Math.random()-0.5)*0.6;
    pred.mode = 'rôde';
    pred.x = Math.max(5, Math.min(W-5, pred.x + pred.vx));
    pred.y = Math.max(5, Math.min(H-5, pred.y + pred.vy));
    if (Math.hypot(pred.vx, pred.vy) > 0.05) pred.angle = Math.atan2(pred.vy, pred.vx);
    return;
  }

  // Vitesse : base + bonus selon l'audace moyenne de la population vivante
  // (population plus téméraire en moyenne → prédateur légèrement plus rapide)
  const avgBold = aliveFishes.reduce((s,f)=>s+f.temperament.boldness,0)/aliveFishes.length;
  const targetSpeed = PRED_BASE_SPEED * (0.85 + avgBold*0.35);
  pred.speed = pred.speed*0.95 + targetSpeed*0.05;

  // Ciblage avec hystérésis
  let nearest=null, minD=Infinity;
  aliveFishes.forEach(f => { const d=dist(pred.x,pred.y,f.x,f.y); if (d<minD){minD=d;nearest=f;} });
  if (!nearest) return;

  let target = pred.target && pred.target.alive ? pred.target : null;
  if (!target) { target = nearest; pred.prevTx = pred.prevTy = null; }
  else {
    const dCur = dist(pred.x,pred.y,target.x,target.y);
    if (nearest!==target && dist(pred.x,pred.y,nearest.x,nearest.y) < dCur*0.6) {
      target = nearest; pred.prevTx = pred.prevTy = null;
    } else minD = dCur;
  }
  pred.target = target;

  if (minD < AGGRO_R) {
    pred.mode = 'chasse';
    pred.chaseTicks++;
    pred.ambushTicks = 0;

    let tx=target.x, ty=target.y;
    if (pred.prevTx !== null) {
      const h = Math.min(minD/Math.max(pred.speed*1.5,0.1), 18);
      tx += (tx-pred.prevTx)*h; ty += (ty-pred.prevTy)*h;
      tx = Math.max(5, Math.min(W-5, tx)); ty = Math.max(5, Math.min(H-5, ty));
    }
    pred.prevTx = target.x; pred.prevTy = target.y;

    const dx=tx-pred.x, dy=ty-pred.y;
    const dn = Math.max(Math.hypot(dx,dy),1);
    pred.vx = pred.vx*0.50 + (dx/dn)*pred.speed*0.50;
    pred.vy = pred.vy*0.50 + (dy/dn)*pred.speed*0.50;

    if (minD > ABANDON_R || pred.chaseTicks > MAX_CHASE_TICKS) {
      pred.mode = 'embuscade';
      pred.chaseTicks = 0;
      pred.prevTx = pred.prevTy = null;
      const pcx = aliveFishes.reduce((s,f)=>s+f.x,0)/aliveFishes.length;
      const pcy = aliveFishes.reduce((s,f)=>s+f.y,0)/aliveFishes.length;
      pred.ambushX = (pred.x+pcx)/2; pred.ambushY = (pred.y+pcy)/2;
      pred.ambushTicks = 0;
    }
  } else if (pred.mode === 'embuscade') {
    pred.ambushTicks++;
    const dx=pred.ambushX-pred.x, dy=pred.ambushY-pred.y;
    const dn = Math.max(Math.hypot(dx,dy),1);
    if (dn>10) {
      pred.vx = pred.vx*0.7 + (dx/dn)*pred.speed*0.12;
      pred.vy = pred.vy*0.7 + (dy/dn)*pred.speed*0.12;
    } else { pred.vx*=0.8; pred.vy*=0.8; }
    if (pred.ambushTicks > 90+Math.random()*60) {
      pred.mode='rôde'; pred.ambushTicks=0; pred.prevTx=pred.prevTy=null;
    }
  } else {
    pred.mode = 'rôde';
    pred.chaseTicks=0; pred.prevTx=pred.prevTy=null;
    const pcx = aliveFishes.reduce((s,f)=>s+f.x,0)/aliveFishes.length;
    const pcy = aliveFishes.reduce((s,f)=>s+f.y,0)/aliveFishes.length;
    const dx=pcx-pred.x, dy=pcy-pred.y;
    const dn = Math.max(Math.hypot(dx,dy),1);
    pred.vx = pred.vx*0.92 + (dx/dn)*pred.speed*0.22 + (Math.random()-0.5)*0.3;
    pred.vy = pred.vy*0.92 + (dy/dn)*pred.speed*0.22 + (Math.random()-0.5)*0.3;
  }

  const pspd = Math.hypot(pred.vx,pred.vy);
  if (pspd > pred.speed*1.6) { pred.vx*=pred.speed*1.6/pspd; pred.vy*=pred.speed*1.6/pspd; }
  pred.x = Math.max(5, Math.min(W-5, pred.x+pred.vx));
  pred.y = Math.max(5, Math.min(H-5, pred.y+pred.vy));
  if (pspd>0.05) pred.angle = Math.atan2(pred.vy,pred.vx);
}

// ══════════════════════════════════════════════════════════════
//  Poisson — décision, physique, faim, peur
// ══════════════════════════════════════════════════════════════
function stepFish(fish, aliveFishes) {
  const t = fish.temperament;
  fish.stepsSurvived++;
  fish.age++;
  fish.tailPhase += 0.16;
  if (fish.reproCooldown > 0) fish.reproCooldown--;

  // ── Métabolisme : la faim baisse plus ou moins vite selon le tempérament ──
  // metabolism=0 (économe) → décroissance ×0.7 ; metabolism=1 (glouton) → ×1.35
  const hungerDec = (1.0/(2500*0.75)) * (0.7 + t.metabolism*0.65);
  fish.energy = Math.max(0, fish.energy - hungerDec);
  if (fish.energy <= 0) {
    fish.alive = false;
    fish.deathCause = 'starvation';
    stats.deaths.starvation++;
    return;
  }

  // ── Perception nourriture (avec bruit selon perception) ───────────────────
  const aliveFoods = foods.filter(f=>!f.eaten);
  const perceptionNoise = (1-t.perception) * 0.35; // 0 = parfait, 0.35 = bruit fort

  let fdDx=0, fdDy=0, fdDist=1;
  if (aliveFoods.length>0) {
    let closest=null, minD2=Infinity;
    aliveFoods.forEach(f=>{ const d2=dist(fish.x,fish.y,f.x,f.y); if(d2<minD2){minD2=d2;closest=f;} });
    fdDx = (closest.x-fish.x)/W + (Math.random()-0.5)*perceptionNoise;
    fdDy = (closest.y-fish.y)/H + (Math.random()-0.5)*perceptionNoise;
    fdDist = Math.min(minD2/(W*0.5),1) + (Math.random()-0.5)*perceptionNoise*0.5;
    fdDist = Math.max(0, Math.min(1, fdDist));
  }

  // ── Perception prédateur (même bruit) ──────────────────────────────────────
  let pdDx=0, pdDy=0, pdDist=1, minPD=Infinity, closestPred=null;
  preds.forEach(p=>{ const d=dist(fish.x,fish.y,p.x,p.y); if(d<minPD){minPD=d;closestPred=p;} });
  if (closestPred) {
    pdDx = (closestPred.x-fish.x)/W + (Math.random()-0.5)*perceptionNoise;
    pdDy = (closestPred.y-fish.y)/H + (Math.random()-0.5)*perceptionNoise;
    pdDist = Math.min(minPD/(W*0.5),1) + (Math.random()-0.5)*perceptionNoise*0.5;
    pdDist = Math.max(0, Math.min(1, pdDist));
  }

  // ── Murs ────────────────────────────────────────────────────────────────
  const wx = Math.min(fish.x, W-fish.x)/(W*0.5);
  const wy = Math.min(fish.y, H-fish.y)/(H*0.5);

  // ── Input social : direction/densité des voisins proches ──────────────────
  // Pondéré côté PERCEPTION par socialPull dans l'input lui-même : un poisson
  // solitaire reçoit un signal social atténué (il "n'y prête pas attention"),
  // un grégaire reçoit le signal plein.
  let socDx=0, socDy=0, neighbors=0;
  aliveFishes.forEach(o=>{
    if (o===fish) return;
    const d = dist(fish.x,fish.y,o.x,o.y);
    if (d < 160 && d > 0.001) { socDx += (o.x-fish.x)/d; socDy += (o.y-fish.y)/d; neighbors++; }
  });
  let socSignal = 0;
  if (neighbors>0) {
    socDx/=neighbors; socDy/=neighbors;
    socSignal = Math.hypot(socDx,socDy);
  }
  const socWeight = 0.2 + t.socialPull*0.8; // solitaire perçoit 20%, grégaire 100%
  const socInputX = socDx * socWeight;
  const socInputY = socDy * socWeight;

  // ── Danger memory ───────────────────────────────────────────────────────
  const isPredActive = closestPred && closestPred.active;
  if (isPredActive && minPD < 260) {
    fish.dangerMem = Math.min(1, fish.dangerMem + 0.18*(1-pdDist));
  } else {
    fish.dangerMem = Math.max(0, fish.dangerMem*0.97);
  }

  // ── Urgence faim + signal temporel ─────────────────────────────────────────
  const hungerUrgency = Math.max(0, 1.0 - fish.energy*2.5);
  fish.timeSinceMeal = Math.min(MEAL_SAT_TICKS, fish.timeSinceMeal+1);
  const mealSignal = fish.timeSinceMeal/MEAL_SAT_TICKS;

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
    Math.hypot(socInputX, socInputY) * Math.sign(socDx||1)*0, // placeholder removed below
  ];
  // Le 16e input encode la direction sociale projetée (vers le groupe = positif)
  // sous forme d'un signal scalaire signé combinant x et y — on garde le réseau
  // à 4 sorties (haut/bas/gauche/droite différentiel), donc on injecte le vecteur
  // social comme DEUX composantes en réutilisant la dernière entrée pour la norme
  // et en biaisant directement ax/ay après le forward (cf. plus bas), ce qui est
  // plus simple et plus interprétable qu'un 17e input.
  inputs[15] = socSignal; // densité/force du groupe perçue [0, ~1]

  const out = mlpForward(fish.weights, inputs);
  let ax = (out[3]-out[2])*FISH_SPD;
  let ay = (out[0]-out[1])*FISH_SPD;

  // ── Influence sociale directe sur l'accélération ───────────────────────────
  // Le réseau reçoit la FORCE du signal social (input 15) et peut apprendre à
  // y réagir, mais la DIRECTION (vers le centroïde des voisins) est appliquée
  // comme un terme physique pondéré par socialPull — un grégaire est
  // mécaniquement tiré vers le groupe, un solitaire ne l'est presque pas.
  if (neighbors>0) {
    ax += socDx * socWeight * 0.9;
    ay += socDy * socWeight * 0.9;
  }

  // ── Répulsion mur ───────────────────────────────────────────────────────
  function wallRep(pos, lo, hi, zone=BORDER_ZONE, force=WALL_FORCE, exp=WALL_EXP) {
    let r=0; const dlo=pos-lo, dhi=hi-pos;
    if (dlo<zone) r += force*(1-dlo/zone)**exp;
    if (dhi<zone) r -= force*(1-dhi/zone)**exp;
    return r;
  }
  ax += wallRep(fish.x,5,W-5);
  ay += wallRep(fish.y,5,H-5);

  // ── Réflexe panique (distance modulée par boldness) ─────────────────────
  // anxieux (boldness=0) panique dès 130px, téméraire (boldness=1) seulement à 70px
  const panicDist = PANIC_DIST_BASE * (1.3 - t.boldness*0.6);
  if (isPredActive && minPD < panicDist) {
    const dxp=fish.x-closestPred.x, dyp=fish.y-closestPred.y;
    const d = Math.max(Math.hypot(dxp,dyp),1);
    ax = (dxp/d)*FISH_SPD*1.6;
    ay = (dyp/d)*FISH_SPD*1.6;
  }

  // ── Vitesse max modulée par métabolisme : glouton légèrement plus rapide ──
  const maxSpd = MAX_SPD * (0.92 + t.metabolism*0.16);

  fish.vx = fish.vx*0.50 + ax*0.50;
  fish.vy = fish.vy*0.50 + ay*0.50;
  const spd = Math.hypot(fish.vx,fish.vy);
  if (spd>maxSpd) { fish.vx*=maxSpd/spd; fish.vy*=maxSpd/spd; }

  const prevX=fish.x, prevY=fish.y;
  fish.x = Math.max(5, Math.min(W-5, fish.x+fish.vx));
  fish.y = Math.max(5, Math.min(H-5, fish.y+fish.vy));
  const moved = Math.hypot(fish.x-prevX, fish.y-prevY);
  fish.distanceTraveled += moved;
  fish.tailAmp = Math.min(1, fish.tailAmp*0.88 + moved*0.18);
  if (Math.abs(fish.vx)>0.01||Math.abs(fish.vy)>0.01) fish.angle = Math.atan2(fish.vy,fish.vx);

  fish.trail.push({x:fish.x,y:fish.y});
  if (fish.trail.length>20) fish.trail.shift();

  // ── Manger ──────────────────────────────────────────────────────────────
  // gain légèrement supérieur pour les gloutons (compense leur métabolisme rapide)
  const hungerGain = HUNGER_GAIN * (0.9 + t.metabolism*0.25);
  aliveFoods.forEach(f=>{
    if (!f.eaten && dist(fish.x,fish.y,f.x,f.y) < FISH_R+FOOD_R) {
      fish.foodEaten++;
      fish.energy = Math.min(1.0, fish.energy+hungerGain);
      fish.mealIntervals.push(fish.timeSinceMeal);
      if (fish.mealIntervals.length>30) fish.mealIntervals.shift();
      fish.timeSinceMeal = 0;
      f.eaten = true; // sera respawné ailleurs par la régulation écologique
    }
  });

  // ── Mort par prédateur ──────────────────────────────────────────────────
  preds.forEach(pred=>{
    if (pred.active && dist(fish.x,fish.y,pred.x,pred.y) < FISH_R+PRED_R) {
      fish.alive = false;
      fish.deathCause = 'predator';
      stats.deaths.predator++;
    }
  });

  // ── Peur (montée modulée par boldness) ──────────────────────────────────
  const fearGain = 0.28 * (1.4 - t.boldness*0.8); // anxieux monte ~1.4×, téméraire ~0.6×
  if (isPredActive && minPD < 200) {
    fish.fear = Math.min(1, fish.fear + fearGain*(1-pdDist));
  } else {
    fish.fear = Math.max(0, fish.fear - 0.04);
  }
  fish.fearAccum += fish.fear;
  fish.fearAvg = fish.fearAccum/Math.max(fish.stepsSurvived,1);
}

// ══════════════════════════════════════════════════════════════
//  Reproduction sexuée
// ══════════════════════════════════════════════════════════════
function handleReproduction() {
  const candidates = fishes.filter(f =>
    f.alive && f.age > MATURITY_AGE && f.energy > REPRO_ENERGY && f.reproCooldown <= 0
  );
  if (candidates.length < 2) return;

  const paired = new Set();
  for (let i=0;i<candidates.length;i++) {
    const a = candidates[i];
    if (paired.has(a.id)) continue;
    for (let j=i+1;j<candidates.length;j++) {
      const b = candidates[j];
      if (paired.has(b.id)) continue;
      if (dist(a.x,a.y,b.x,b.y) < REPRO_RANGE) {
        // Reproduction !
        paired.add(a.id); paired.add(b.id);
        const childWeights = mutateWeights(crossoverWeights(a.weights,b.weights), MUT_STD);
        const childTemp    = mutateTemperament(crossoverTemperament(a.temperament,b.temperament));
        const childGen = Math.max(a.gen,b.gen)+1;
        genCounter = Math.max(genCounter, childGen);

        a.energy -= REPRO_COST; b.energy -= REPRO_COST;
        a.reproCooldown = REPRO_COOLDOWN; b.reproCooldown = REPRO_COOLDOWN;
        a.children++; b.children++;

        fishes.push(makeFish({
          weights: childWeights,
          temperament: childTemp,
          gen: childGen,
          x: (a.x+b.x)/2 + (Math.random()-0.5)*10,
          y: (a.y+b.y)/2 + (Math.random()-0.5)*10,
          energy: 0.55,
          parents: [a.id,b.id],
        }));
        stats.births++;
        break;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  Interprétation comportementale (dashboard)
// ══════════════════════════════════════════════════════════════
function behaviourInsight(fish) {
  const alive = fish.alive;
  const fe = fish.foodEaten;
  const steps = fish.stepsSurvived;
  const fear = fish.fearAvg;
  const t = fish.temperament;

  if (!alive) {
    if (fish.deathCause === 'starvation') {
      if (fe === 0) return "N'a jamais réussi à se nourrir — stratégie non viable.";
      if (fear > 0.35) return "La peur a dominé : recherche de nourriture trop prudente.";
      return "A mangé, mais pas assez régulièrement — épuisement progressif.";
    }
    if (fish.deathCause === 'predator') {
      if (fe === 0) return "Tué sans avoir mangé — n'a pas eu le temps d'apprendre.";
      return `A survécu ${steps} ticks et eu ${fish.children} descendant(s) avant d'être attrapé.`;
    }
    if (fish.deathCause === 'old_age') {
      return `Mort de vieillesse après ${steps} ticks — ${fish.children} descendant(s). Belle longévité.`;
    }
  }

  if (steps < 100) return "Phase d'exploration initiale du génome.";

  const foodPerTick = fe/Math.max(steps,1);
  if (fish.mealIntervals.length >= 4) {
    const avgGap = fish.mealIntervals.reduce((a,b)=>a+b,0)/fish.mealIntervals.length;
    if (avgGap < 120 && fear < 0.3) return "✓ Rythme alimentaire régulier — pression de faim maîtrisée.";
    if (avgGap > 300) return "Repas trop espacés — la faim grimpe dangereusement entre deux prises.";
  }

  if (fish.children > 0 && fish.energy > REPRO_ENERGY*0.8)
    return `✓ Stratégie reproductive efficace — ${fish.children} descendant(s), bonne réserve d'énergie.`;
  if (foodPerTick > 0.015 && fear < 0.2)
    return "✓ Excellente stratégie alimentaire, peu de peur inutile.";
  if (foodPerTick < 0.003 && fear > 0.4)
    return "Paralysé par la peur — survie passive, la faim arrive.";
  if (t.socialPull > 0.65 && neighborCountOf(fish) >= 2)
    return "Profil grégaire — suit le groupe, profite de la vigilance collective.";
  if (t.socialPull < 0.35)
    return "Profil solitaire — explore seul, moins de compétition directe.";

  return "Comportement stable, en cours d'optimisation.";
}
function neighborCountOf(fish) {
  let n=0;
  fishes.forEach(o=>{ if(o!==fish && o.alive && dist(fish.x,fish.y,o.x,o.y)<160) n++; });
  return n;
}

// ══════════════════════════════════════════════════════════════
//  Décor statique (identique v5)
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

  ctx.save(); ctx.globalAlpha=0.022;
  for(let i=0;i<10;i++){
    const t2=tick*.002+i*.55;
    const x=(0.5+.5*Math.sin(t2*1.1))*cw, y=(0.15+.1*Math.sin(t2*.9+1))*ch;
    const r=(18+10*Math.sin(t2*.7))*(cw/800);
    const gc=ctx.createRadialGradient(x,y,0,x,y,r);
    gc.addColorStop(0,'#3ab4ff'); gc.addColorStop(1,'rgba(58,180,255,0)');
    ctx.fillStyle=gc; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();

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
//  Prédateur requin
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

    ctx.save(); ctx.rotate(tailSway);
    ctx.beginPath();
    ctx.moveTo(-L*.55,0); ctx.lineTo(-L,-L*.36); ctx.lineTo(-L*.95,0); ctx.lineTo(-L,L*.36);
    ctx.closePath();
    ctx.fillStyle=isChasing?'#991111':isActive?'#5a1515':'#2d0d0d'; ctx.fill();
    ctx.restore();

    ctx.beginPath();
    ctx.moveTo(L,0);
    ctx.bezierCurveTo(L*.6,-L*.22,-L*.2,-L*.25,-L*.55,-L*.04);
    ctx.bezierCurveTo(-L*.6,-L*.02,-L*.6,L*.02,-L*.55,L*.04);
    ctx.bezierCurveTo(-L*.2,L*.25,L*.6,L*.22,L,0);
    ctx.fillStyle=isChasing?'#b21515':isActive?'#4a1212':'#2a0d0d'; ctx.fill();

    ctx.beginPath();
    ctx.moveTo(L*.7,0);ctx.bezierCurveTo(L*.4,-L*.1,-L*.1,-L*.1,-L*.28,-L*.02);
    ctx.bezierCurveTo(-L*.28,L*.02,-L*.1,L*.1,L*.4,L*.1); ctx.closePath();
    ctx.fillStyle=isChasing?'#cc2a2a44':'#3d181844'; ctx.fill();

    ctx.beginPath();
    ctx.moveTo(L*.06,-L*.2);ctx.lineTo(L*.26,-L*.5);ctx.lineTo(L*.4,-L*.22);
    ctx.closePath();
    ctx.fillStyle=isChasing?'#991111':isActive?'#3d1818':'#2a0d0d'; ctx.fill();

    ctx.beginPath();
    ctx.moveTo(L*.2,L*.15);ctx.lineTo(L*.05,L*.42);ctx.lineTo(-L*.12,L*.18);
    ctx.closePath();
    ctx.fillStyle=isActive?'#aa1818':'#300d0d'; ctx.fill();

    ctx.shadowBlur=0;
    ctx.beginPath();ctx.arc(L*.55,-L*.07,L*.07,0,Math.PI*2);
    ctx.fillStyle='#080808'; ctx.fill();
    ctx.beginPath();ctx.arc(L*.55,-L*.07,L*.03,0,Math.PI*2);
    ctx.fillStyle=isChasing?'#ff1010':isAmbush?'#ff8800':'#333'; ctx.fill();

    if(isChasing){
      ctx.globalAlpha=0.65;
      for(let tt=0;tt<3;tt++){
        const tx2=L*.87-tt*L*.09, tw=L*.04;
        ctx.beginPath();ctx.moveTo(tx2,-L*.03);ctx.lineTo(tx2-tw,-L*.1);ctx.lineTo(tx2-tw*2,-L*.03);
        ctx.fillStyle='#f0f0f0'; ctx.fill();
      }
    }

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

  if(fish.alive && fish.trail && fish.trail.length>3){
    ctx.save();
    for(let i=1;i<fish.trail.length;i++){
      const ti=i/fish.trail.length;
      ctx.beginPath();
      ctx.moveTo(fish.trail[i-1].x*scX, fish.trail[i-1].y*scY);
      ctx.lineTo(fish.trail[i].x*scX,   fish.trail[i].y*scY);
      ctx.strokeStyle=fish.color;
      ctx.globalAlpha=ti*0.065*(fish.fear>0.35?1.6:1);
      ctx.lineWidth=ti*2.2; ctx.stroke();
    }
    ctx.restore();
  }

  ctx.save(); ctx.translate(sx,sy); ctx.rotate(fish.angle||0);
  const alive=fish.alive;
  ctx.globalAlpha=alive?1:.16;

  // Taille croît légèrement avec l'âge jusqu'à maturité (effet "croissance")
  const growth = Math.min(1, fish.age/MATURITY_AGE);
  const L = 7 + growth*4; // 7px (juvénile) → 11px (adulte)

  const tailSway=alive?Math.sin(fish.tailPhase||0)*(fish.tailAmp||0)*0.75:0;

  let bodyColor=fish.color;
  if(alive && fish.energy < 0.35){
    const tt=1-(fish.energy/0.35);
    bodyColor=lerpColor(fish.color,'#888855',tt*0.55);
  }

  const fearGlow=fish.fear>0.4;
  if(alive){
    ctx.shadowBlur=fearGlow?16:8;
    ctx.shadowColor=fearGlow?'#ff4444':bodyColor;
  }

  ctx.save(); ctx.translate(-L*1.1,0); ctx.rotate(tailSway*.7);
  ctx.beginPath();
  ctx.moveTo(0,0);ctx.lineTo(-L*.72,-L*.68);ctx.lineTo(-L*.38,0);ctx.lineTo(-L*.72,L*.68);
  ctx.closePath(); ctx.fillStyle=bodyColor+'bb'; ctx.fill();
  ctx.restore();

  ctx.beginPath();
  ctx.moveTo(L*1.2,0);
  ctx.bezierCurveTo(L*.8,-L*.6,-L*.5,-L*.68,-L*1.1,0);
  ctx.bezierCurveTo(-L*.5,L*.68,L*.8,L*.6,L*1.2,0);
  ctx.fillStyle=bodyColor; ctx.fill();

  ctx.beginPath();
  ctx.moveTo(L*.9,0);ctx.bezierCurveTo(L*.5,-L*.27,-L*.2,-L*.28,-L*.68,0);
  ctx.bezierCurveTo(-L*.2,L*.28,L*.5,L*.27,L*.9,0);
  ctx.fillStyle='rgba(255,255,255,0.11)'; ctx.fill();

  ctx.beginPath();
  ctx.moveTo(-L*.28,-L*.48);ctx.quadraticCurveTo(L*.1,-L*.98,L*.48,-L*.5);
  ctx.lineTo(L*.48,-L*.37);ctx.quadraticCurveTo(L*.1,-L*.68,-L*.28,-L*.37);
  ctx.closePath(); ctx.globalAlpha=alive?.68:.14;
  ctx.fillStyle=bodyColor+'aa'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  ctx.beginPath();
  ctx.moveTo(-L*.08,L*.40);ctx.quadraticCurveTo(L*.14,L*.82,L*.44,L*.43);
  ctx.lineTo(L*.44,L*.33);ctx.quadraticCurveTo(L*.14,L*.58,-L*.08,L*.30);
  ctx.closePath(); ctx.globalAlpha=alive?.52:.1;
  ctx.fillStyle=bodyColor+'88'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  ctx.shadowBlur=0;
  ctx.beginPath();ctx.moveTo(L*.38,L*.17);ctx.quadraticCurveTo(L*.68,L*.52,L*.08,L*.48);
  ctx.closePath(); ctx.globalAlpha=alive?.48:.08;
  ctx.fillStyle=bodyColor+'77'; ctx.fill(); ctx.globalAlpha=alive?1:.16;

  ctx.shadowBlur=0;
  ctx.beginPath();ctx.arc(L*.54,-L*.14,L*.21,0,Math.PI*2);
  ctx.fillStyle='#0c190c'; ctx.fill();
  ctx.beginPath();ctx.arc(L*.54,-L*.14,L*.13,0,Math.PI*2);
  ctx.fillStyle=alive?'#ffffff':'#333'; ctx.globalAlpha=alive?.88:.28; ctx.fill();
  ctx.globalAlpha=alive?1:.16;
  ctx.beginPath();ctx.arc(L*.58,-L*.17,L*.055,0,Math.PI*2);
  ctx.fillStyle='#090909'; ctx.fill();

  if(alive){
    ctx.globalAlpha=0.07; ctx.strokeStyle='#fff'; ctx.lineWidth=0.55;
    for(let i=0;i<3;i++){
      ctx.beginPath();ctx.arc(-L*.18+i*L*.44,0,L*.28-i*.01,-Math.PI*.5,Math.PI*.5);ctx.stroke();
    }
    ctx.globalAlpha=1;
  }

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

  if(alive){
    ctx.shadowBlur=0;
    const bw=L*2.3;
    const h=fish.energy;
    const hColor = h>0.5 ? '#f0c030' : h>0.25 ? '#e08020' : '#cc2020';
    ctx.globalAlpha=Math.max(0, 0.85 - h*0.5);
    ctx.fillStyle=hColor;
    ctx.fillRect(-bw/2,-L*1.75,bw*h,2);
    ctx.strokeStyle='rgba(255,255,255,.08)';ctx.lineWidth=.4;
    ctx.strokeRect(-bw/2,-L*1.75,bw,2);
    ctx.globalAlpha=1;
  }

  // Marqueur de maturité reproductive (petit point vert si prêt à se reproduire)
  if(alive && fish.age>MATURITY_AGE && fish.energy>REPRO_ENERGY && fish.reproCooldown<=0){
    ctx.shadowBlur=4; ctx.shadowColor='#39ff8a';
    ctx.beginPath(); ctx.arc(0, -L*1.95, 1.6, 0, Math.PI*2);
    ctx.fillStyle='#39ff8a'; ctx.fill();
    ctx.shadowBlur=0;
  }

  ctx.restore();

  if(hoveredFish===fish){
    ctx.save(); ctx.globalAlpha=.9;
    ctx.fillStyle=fish.color; ctx.font='10px JetBrains Mono,monospace';
    ctx.fillText(`#${fish.name||fish.id} g${fish.gen}`,sx+15,sy-15); ctx.restore();
  }
}

function lerpColor(c1, c2, t) {
  // Supporte hex (#rrggbb) — temperamentColor renvoie du hsl(), donc on
  // convertit via un canvas 1x1 pour rester générique.
  const a = colorToRgb(c1), b = colorToRgb(c2);
  const r=Math.round(a[0]+(b[0]-a[0])*t), g=Math.round(a[1]+(b[1]-a[1])*t), bl=Math.round(a[2]+(b[2]-a[2])*t);
  return `rgb(${r},${g},${bl})`;
}
const _colorCache = {};
function colorToRgb(c) {
  if (_colorCache[c]) return _colorCache[c];
  const d = document.createElement('canvas').getContext('2d');
  d.fillStyle = c; d.fillRect(0,0,1,1);
  const data = d.getImageData(0,0,1,1).data;
  const rgb = [data[0],data[1],data[2]];
  _colorCache[c] = rgb;
  return rgb;
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
  fishes.forEach(f => { if (f.alive) drawFish(f, scX, scY); });
}

// ══════════════════════════════════════════════════════════════
//  UI / Dashboard — vue population (remplace les cartes par lignée)
// ══════════════════════════════════════════════════════════════
function buildLineageCards() {
  // Conservé pour compat de nommage avec le HTML existant, mais la
  // structure est reconstruite dynamiquement à chaque updateUI car la
  // population fluctue (naissances/morts).
  const container = document.getElementById('lineageCards');
  if (container) container.innerHTML = '<div style="font-size:9px;color:var(--muted)">Population en cours d\'initialisation…</div>';
}

function traitBar(label, value, lowLabel, highLabel) {
  const pct = Math.round(value*100);
  return `
    <div class="stat-row" style="margin-bottom:1px">
      <span style="width:70px">${label}</span>
      <div class="hunger-bar-bg" style="flex:1;margin:0 5px">
        <div class="hunger-bar-fill" style="width:${pct}%;background:#00b4d8"></div>
      </div>
      <span style="font-size:8px;color:var(--muted);width:60px;text-align:right">${pct<50?lowLabel:highLabel}</span>
    </div>`;
}

function updateUI(force=false) {
  const alive = fishes.filter(f=>f.alive);
  const pop = alive.length;

  document.getElementById('gTick').textContent = tick;
  document.getElementById('gAlive').textContent = `${pop} (max gén. ${genCounter})`;
  document.getElementById('gFood').textContent  = foods.filter(f=>!f.eaten).length + '/' + foods.length;
  const pred = preds[0];
  if (pred) {
    const modeLabels = {'chasse':'⚡ chasse','embuscade':'👁 embuscade','rôde':'~ rôde'};
    document.getElementById('gPred').textContent =
      pred.active ? (modeLabels[pred.mode]||pred.mode) + ` (v${pred.speed.toFixed(2)})` : '· inactif';
  }

  const container = document.getElementById('lineageCards');
  if (!container) return;

  // ── Stats globales de population ──────────────────────────────────────
  const avgEnergy = pop ? alive.reduce((s,f)=>s+f.energy,0)/pop : 0;
  const avgAge    = pop ? alive.reduce((s,f)=>s+f.age,0)/pop : 0;
  const avgTemp = {perception:0,metabolism:0,socialPull:0,boldness:0};
  alive.forEach(f=>{ for(const k in avgTemp) avgTemp[k]+=f.temperament[k]; });
  for (const k in avgTemp) avgTemp[k] = pop ? avgTemp[k]/pop : 0;

  let html = `
    <div class="lineage-card" style="border-left-color:#00b4d8">
      <div class="lineage-header">
        <span class="lineage-name" style="color:#00b4d8">Population</span>
        <span class="lineage-status">${pop} vivants</span>
      </div>
      <div class="stat-row"><span>naissances cumul.</span><span class="stat-val">${stats.births}</span></div>
      <div class="stat-row"><span>morts (faim)</span><span class="stat-val">${stats.deaths.starvation}</span></div>
      <div class="stat-row"><span>morts (prédateur)</span><span class="stat-val">${stats.deaths.predator}</span></div>
      <div class="stat-row"><span>morts (vieillesse)</span><span class="stat-val">${stats.deaths.old_age}</span></div>
      <div class="stat-row"><span>énergie moy.</span><span class="stat-val">${Math.round(avgEnergy*100)}%</span></div>
      <div class="stat-row"><span>âge moyen</span><span class="stat-val">${Math.round(avgAge)}t</span></div>
      <div class="section-title" style="margin-top:8px">tempérament moyen (pop.)</div>
      ${traitBar('perception', avgTemp.perception, 'myope', 'clairvoyant')}
      ${traitBar('métabolisme', avgTemp.metabolism, 'économe', 'glouton')}
      ${traitBar('social', avgTemp.socialPull, 'solitaire', 'grégaire')}
      ${traitBar('hardiesse', avgTemp.boldness, 'anxieux', 'téméraire')}
      <canvas class="sparkline" id="spark-pop" height="26"></canvas>
    </div>
  `;

  // ── Individus (vivants, triés par énergie décroissante, max 12 affichés) ──
  const sorted = alive.slice().sort((a,b)=>b.energy-a.energy).slice(0,12);
  sorted.forEach(fish => {
    const tags = [];
    if (fish.age>MATURITY_AGE && fish.energy>REPRO_ENERGY && fish.reproCooldown<=0)
      tags.push({text:'PRÊT À SE REPRODUIRE',color:'#39ff8a'});
    if (fish.fear>0.4) tags.push({text:'PEUR ÉLEVÉE',color:'#e74c3c'});
    const tagHTML = tags.map(tg=>`<span class="behav-tag" style="color:${tg.color};border-color:${tg.color}33">${tg.text}</span>`).join(' ');

    html += `
      <div class="lineage-card" style="--lcolor:${fish.color}">
        <div class="lineage-header">
          <span class="lineage-name" style="color:${fish.color}">#${fish.id} <span style="font-size:9px;color:var(--muted)">gén.${fish.gen}</span></span>
          <span class="lineage-status">${temperamentLabel(fish.temperament)}</span>
        </div>
        <div class="stat-row"><span>âge</span><span class="stat-val">${fish.age}t</span></div>
        <div class="stat-row"><span>nourriture</span><span class="stat-val">${fish.foodEaten}</span></div>
        <div class="stat-row"><span>descendants</span><span class="stat-val">${fish.children}</span></div>
        <div class="hunger-bar-wrap">
          <span style="font-size:9px;color:var(--muted)">énergie</span>
          <div class="hunger-bar-bg">
            <div class="hunger-bar-fill" style="width:${Math.round(fish.energy*100)}%;background:${fish.energy>0.5?'#f0c030':fish.energy>0.25?'#e08020':'#cc2020'}"></div>
          </div>
          <span class="hunger-label">${Math.round(fish.energy*100)}%</span>
        </div>
        ${tagHTML}
        <div class="behav-insight">${behaviourInsight(fish)}</div>
      </div>
    `;
  });

  container.innerHTML = html;
  requestAnimationFrame(drawPopSparkline);
}

function drawPopSparkline() {
  const cvs = document.getElementById('spark-pop');
  if (!cvs) return;
  cvs.width = cvs.offsetWidth; cvs.height = 26;
  const c = cvs.getContext('2d');
  const data = stats.popHistory.map(h=>h.pop);
  if (data.length<2) return;
  const mx = Math.max(...data, 1);
  const cw=cvs.width, ch=cvs.height;
  c.clearRect(0,0,cw,ch);
  const grad=c.createLinearGradient(0,0,0,ch);
  grad.addColorStop(0,'#00b4d82a'); grad.addColorStop(1,'transparent');
  c.fillStyle=grad; c.beginPath();
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*cw, y=ch-(v/mx)*ch*.86-2;
    i===0?c.moveTo(x,ch):undefined;
    c.lineTo(x,y);
  });
  c.lineTo(cw,ch); c.closePath(); c.fill();
  c.strokeStyle='#00b4d8'; c.lineWidth=1.4;
  c.shadowBlur=5; c.shadowColor='#00b4d8'; c.globalAlpha=.82;
  c.beginPath();
  data.forEach((v,i)=>{
    const x=(i/(data.length-1))*cw, y=ch-(v/mx)*ch*.86-2;
    i===0?c.moveTo(x,y):c.lineTo(x,y);
  });
  c.stroke();
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
    const t=f.temperament;
    tooltip.innerHTML=`
      <b style="color:${f.color}">#${f.id}</b> · gén. ${f.gen} · ${temperamentLabel(t)}<br>
      âge : <b>${f.age}t</b> · énergie : <b>${Math.round(f.energy*100)}%</b><br>
      nourriture : <b>${f.foodEaten}</b> · descendants : <b>${f.children}</b><br>
      depuis dernier repas : ${Math.round(f.timeSinceMeal)}t<br>
      peur : ${Math.round(f.fear*100)}% (moy. ${Math.round(f.fearAvg*100)}%)<br>
      danger memory : ${Math.round(f.dangerMem*100)}%<br>
      perception ${Math.round(t.perception*100)}% · métabolisme ${Math.round(t.metabolism*100)}%<br>
      social ${Math.round(t.socialPull*100)}% · hardiesse ${Math.round(t.boldness*100)}%
      ${f.parents ? `<br>parents : #${f.parents[0]} × #${f.parents[1]}` : ''}
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

function init() {
  generateDecor();
  spawnWorld();
  updateUI(true);
  requestAnimationFrame(loop);
}
init();