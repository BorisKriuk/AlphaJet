# AlphaJet

A conceptual aircraft designer that evolves a 3D jet from a mission specification using a genetic algorithm coupled with an anatomically-disentangled variational autoencoder. The fitness of every candidate design is computed from a closed-form physics model covering aerodynamics, structures, weights, stability, and volumetric packaging.

<p align="center">
  <img src="docs/aircraft.png" alt="AlphaJet evolved design" width="720">
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="flask" src="https://img.shields.io/badge/Flask-2.3+-000000?logo=flask&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## Overview

The user supplies a mission (target gross mass, payload, range, cruise speed, engine count and thrust, structural areal density) and a hard geometric envelope. AlphaJet then evolves a population of aircraft over a fixed number of generations, streaming each generation's geometry and fitness breakdown to a Three.js viewer in the browser.

The objective is not to replace conceptual design tools such as OpenVSP or SUAVE, but to demonstrate that a compact analytical pipeline — anatomical parameters, deterministic voxelization, and a transparent physics model — is sufficient to drive an interactive evolutionary search across realistic aircraft topologies.

---

## Key contributions

**Hybrid GA / AD-VAE search.** A genetic algorithm operates over a 25-dimensional anatomical parameter space (span, sweep, taper, fineness ratio, fin geometry, engine placement, etc.). A small 3D Anatomically-Disentangled VAE is trained on synthetic jets to provide a learned shape prior and a differentiable latent representation.

**Closed-form physics fitness.** Every fitness component is derived analytically rather than learned. The model includes parasite, induced, and wave drag (Korn–Mason), Breguet range, wing-bending stress at the root, longitudinal static margin, V-tail volume coefficient, wing loading, and a fuel-versus-payload volumetric packing check.

**Topology-preserving evolution.** Five tail topologies — conventional, T-tail, cruciform, V-tail, and flying-wing — are seeded round-robin and protected by topology elitism, preventing premature collapse onto a single configuration.

**Asymmetric fuselage modeling.** The fuselage is constructed with a blunter cockpit nose and a small droop, and a sharper tail cone with proper upsweep, rather than the symmetric body-of-revolution typical of toy generators.

**Mount-aware geometry scoring.** Engines and horizontal stabilizers are scored on whether their bounding regions actually intersect the structure they claim to attach to (fuselage, fin, wing pylon). Floating components are penalized in fitness.

**Constructive single-engine layout.** One-engine designs are generated with a rear-mounted nozzle protruding from the upswept tail cone, consistent with business-jet and pusher configurations.

**Hard envelope projection.** The user-defined aircraft envelope and per-engine bounding box are projected onto every individual at every generation, guaranteeing that the final design fits the specified volume.

**Stagnation restarts.** When mean fitness plateaus, half of the population is replaced with fresh seeds drawn across all topologies, preventing the search from settling in narrow local minima.

**Live streaming UI.** A Flask + Socket.IO backend pushes the voxel cloud and a full per-component fitness breakdown to a Three.js viewer after every generation.

---

## Installation

```bash
git clone https://github.com/BorisKriuk/AlphaJet.git
cd AlphaJet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5011` and start the evolution from the UI.

The first run trains the AD-VAE on 4,000 synthetic jets (approximately one minute on CPU, seconds on GPU). Trained weights are cached to `advae.pt` and reused on subsequent runs.

---

## Inputs

| Group       | Field                  | Description                                      |
| ----------- | ---------------------- | ------------------------------------------------ |
| Envelope    | L, H, W                | Hard outer bounding box of the aircraft (m)      |
| Envelope    | Engine L, H, W         | Per-engine bounding box (m)                      |
| Mission     | Gross mass target      | Target maximum take-off weight (kg)              |
| Mission     | Payload target         | Optional desired payload (kg)                    |
| Mission     | Required range         | Mission range (km)                               |
| Mission     | Cruise speed           | True airspeed at cruise (m/s)                    |
| Propulsion  | Number of engines      | 0 for automatic selection, or 1 to 4             |
| Propulsion  | Total engine thrust    | Sea-level static thrust (kN)                     |
| Structure   | Areal density          | Skin mass per square meter (CFRP ~14, Al ~18, Ti ~28) |
| Search      | Generations            | GA iteration budget                              |

---

## Pipeline

```
 ┌────────────────┐  25-D anatomy  ┌────────────────┐  voxels  ┌────────────────┐
 │  GA + AD-VAE   │ ───────────── ▶│ analytical 3D  │ ───────▶ │  physics model │
 │   population   │                │   voxelizer    │          │   + fitness    │
 └───────▲────────┘                └────────────────┘          └────────┬───────┘
         │                                                              │
         └────── tournament + topology elites + mutation  ◀─────────────┘
```

1. The initial population is seeded with an equal mix of all five tail topologies.
2. Each genome is decoded into 25 anatomical parameters.
3. The aircraft is voxelized analytically with the user envelope applied.
4. The physics model returns a fitness score and over thirty sub-metrics.
5. Selection uses fitness-, topology-, and diversity-based elites, with stagnation restarts.

---

## Repository layout

```
AlphaJet/
├── app.py            # Flask + Socket.IO server and main run loop
├── evolution.py      # Genetic algorithm: seeding, repair, selection, decoding
├── physics.py        # Drag, range, weights, mounts, stability, fitness aggregation
├── dataset.py        # Parameter ranges and analytical voxelizer
├── advae.py          # Anatomically-Disentangled 3D VAE
├── train.py          # AD-VAE training on synthetic jets
├── templates/        # Three.js front-end
├── docs/
│   └── aircraft.png  # Sample evolved design
├── requirements.txt
└── README.md
```

---

## Live outputs

The browser viewer reports, for the current best individual:

- Voxelized geometry, with structure and engines rendered as separate components.
- A fitness breakdown including L/D, lift coefficient, Mach versus critical Mach, range ratio, wing-root stress, static margin, mount score, and envelope compliance.
- The full set of 25 evolved anatomical parameters.

---

## Roadmap

- Center-of-gravity and payload visualizer overlay.
- Export of evolved geometry to STEP and STL.
- Multi-objective Pareto front across range, payload, and mass.
- Extended mission profile (climb, loiter, dash) beyond cruise.
- Multi-aircraft co-design.

---

## Disclaimer

AlphaJet is a research and educational tool. The physics model is intentionally analytical and lightweight so that the genetic algorithm can run interactively in the browser. The output is not certified for any engineering or operational use.

---

## License

Released under the MIT License. See `LICENSE` for details.

