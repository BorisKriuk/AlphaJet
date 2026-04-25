# ✈️ AlphaJet

> **Evolve a jet from a box of numbers.**
> Type in a mass, a range, an engine, and a bounding box — AlphaJet evolves a full 3D aircraft that actually closes the loop on aerodynamics, structures, weights, and packaging.

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="flask" src="https://img.shields.io/badge/Flask-2.3+-000000?logo=flask&logoColor=white">
  <img alt="socketio" src="https://img.shields.io/badge/Socket.IO-live-010101?logo=socketdotio&logoColor=white">
  <img alt="status" src="https://img.shields.io/badge/status-experimental-blue">
</p>

---

## 🌟 What is this?

AlphaJet is a **conceptual aircraft designer** that runs entirely in your browser tab.
You give it a *mission* (mass, range, cruise speed, payload, engine count, hard size box) and a **genetic algorithm** evolves a voxel-rendered aircraft in real time, scored by a small but honest physics model.

Watch the wings, fuselage, fins, and engines morph generation by generation until the jet is feasible.

---

## ✨ Novelties

🧬 **GA + AD-VAE hybrid** — A genetic algorithm searches a 25-D anatomical space; a tiny 3D **Anatomically-Disentangled VAE** keeps the latent shape prior coherent.

📐 **Honest physics, not a black box** — Every fitness number is derived: drag (CD₀ / CDᵢ / CDw with Korn–Mason wave drag), Breguet range, wing-bending stress, static margin, V-tail volume coefficient, wing loading, fuel-vs-payload volumetric packing.

🪶 **Five tail topologies, equally explored** — Conventional, T-tail, cruciform, V-tail, and flying-wing are all seeded round-robin and protected by **topology elitism**, so the search never collapses to one shape.

🛩️ **Modern asymmetric fuselage** — Blunter cockpit nose with a small droop, sharper tail cone with proper **upsweep** — no more torpedo bodies.

🔧 **Mount-aware geometry** — Engines and H-tails are scored on whether they actually *touch* the structure they claim to attach to (fuselage, fin, wing pylon). Floating parts get crushed in fitness.

🚀 **Single-engine = rear-mounted by construction** — 1-engine designs always show a visible nozzle protruding from the upswept tail (think pusher-jet / business-jet rear mount).

📦 **Hard size box** — User-set `L × H × W` and per-engine envelope are *projected* onto every individual every generation. The jet always fits.

🔁 **Stagnation restarts** — When fitness plateaus, half the population is replaced with fresh seeds to escape local minima.

🌐 **Live streaming UI** — Flask + Socket.IO push every generation's voxel cloud and fitness breakdown to a Three.js viewer in real time.

---

## 🚀 Quick start

```bash
git clone https://github.com/<you>/AlphaJet.git
cd AlphaJet
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5011** and press **Start Evolution**.

> First run trains the AD-VAE on 4,000 synthetic jets (~1 min on CPU, seconds on GPU). The weights are cached to `advae.pt`.

---

## 🎛️ Inputs

| Group       | Field                                 | Meaning                                    |
| ----------- | ------------------------------------- | ------------------------------------------ |
| Box         | `L × H × W`                           | Hard outer envelope of the aircraft (m)    |
| Box         | `Engine L × H × W`                    | Hard cap on each engine's bounding box (m) |
| Mission     | `Gross mass target`                   | Target MTOW (kg)                           |
| Mission     | `Payload target`                      | Optional desired payload (kg)              |
| Mission     | `Required range`                      | Mission range (km)                         |
| Mission     | `Cruise speed`                        | True airspeed at cruise (m/s)              |
| Propulsion  | `Number of engines`                   | 0 = auto, or force 1–4                     |
| Propulsion  | `Total engine thrust`                 | Sea-level static (kN)                      |
| Structure   | `Areal density`                       | Skin mass per m² (CFRP ≈ 14, Al ≈ 18, Ti ≈ 28) |
| Search      | `Generations`                         | GA budget                                  |

---

## 🧠 How it works

```
 ┌──────────────┐   25-D anatomy   ┌──────────────┐   voxels   ┌──────────────┐
 │  GA  +  AD-VAE  │ ───────────── ▶ │ analytical 3D │ ───────── ▶ │  physics +   │
 │   population    │                 │  voxelizer    │             │   fitness    │
 └──────▲───────┘                    └──────────────┘             └──────┬───────┘
        │                                                                │
        └────────── tournament + topology elites + mutation ◀────────────┘
```

1. **Seed** an equal mix of all 5 tail topologies.
2. **Decode** each genome into 25 anatomical parameters (span, sweep, taper, fineness, …).
3. **Voxelize** the aircraft analytically (no neural decoder error) with hard size caps applied.
4. **Score** with a physics model that returns 30+ sub-metrics.
5. **Select & reproduce** with fitness-, topology-, and diversity-elites, plus stagnation restarts.

---

## 📁 Layout

```
AlphaJet/
├─ app.py            # Flask + Socket.IO server, run loop
├─ evolution.py      # GA: seeding, repair, selection, decode
├─ physics.py        # Drag, range, weights, mounts, stability — fitness
├─ dataset.py        # Param ranges + analytical voxelizer (the "renderer")
├─ advae.py          # Anatomical-Disentangled 3D VAE
├─ train.py          # AD-VAE training on synthetic jets
├─ templates/        # Three.js front-end
└─ requirements.txt
```

---

## 📊 What you see live

- **Voxel viewer** — structure (light blue) and engines (red), with smooth-cube rendering
- **Fitness breakdown** — L/D, CL, Mach vs Mcrit, range ratio, wing root stress, static margin, mount score, block score, …
- **Anatomy readout** — every one of the 25 evolved parameters

---

## 🗺️ Roadmap

- [ ] CG/payload visualizer overlay
- [ ] Export to STEP / STL
- [ ] Pareto front (range vs payload vs mass)
- [ ] Mission profile beyond cruise (climb / loiter / dash)
- [ ] Multi-jet fleet co-design

---

## ⚠️ Disclaimer

AlphaJet is a **conceptual / educational tool**. The physics is intentionally simple — closed-form, fast, differentiable-ish — so a GA can run interactively. Do not fly anything based on the output. 🙂

---

## 📜 License

MIT — do whatever, just don't blame me.
