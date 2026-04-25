import os
import threading
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO

from advae     import ADVAE, denormalize_params, ANATOMICAL_DIM
from evolution import Evolution
from train     import train

MODEL_PATH = 'advae.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'alphajet'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MODEL   = None
_thread = None
_stop   = False

POP_SIZE     = 120
DEFAULT_GENS = 150


def load_or_train_model():
    model = ADVAE().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("[app] loaded AD-VAE from disk.")
        except Exception as e:
            print(f"[app] weight shape mismatch ({e}); retraining.")
            model = train(n_samples=4000, epochs=40, device=device,
                          save_path=MODEL_PATH)
    else:
        print("[app] No model found. Training AD-VAE...")
        model = train(n_samples=4000, epochs=40, device=device,
                      save_path=MODEL_PATH)
    model.eval()
    return model


def voxels_to_points(vox, threshold=0.5):
    return np.argwhere(vox > threshold).astype(int).tolist()


def split_voxels(struct_grid, eng_grid, threshold=0.5):
    """structure-only points + engine-only points"""
    eng_pts = voxels_to_points(eng_grid, threshold)
    structure_only = struct_grid.copy()
    structure_only[eng_grid > threshold] = 0
    struct_pts = voxels_to_points(structure_only, threshold)
    return struct_pts, eng_pts


def run_evolution(user_spec, generations=DEFAULT_GENS):
    global _stop
    _stop = False
    evo = Evolution(MODEL, device=device, pop_size=POP_SIZE)
    pop = evo.initial_population(user_spec)

    best_fit, best_ever = -np.inf, None
    stagnation = 0

    for gen in range(generations):
        if _stop: break
        fits, brks = evo.evaluate(pop, user_spec)
        bi = int(np.argmax(fits))

        if fits[bi] > best_fit + 1e-4:
            best_fit, best_ever, stagnation = float(fits[bi]), pop[bi].copy(), 0
        else:
            stagnation += 1

        struct_grid, eng_grid = evo.decode_voxels_hires(
            pop[bi], user_spec=user_spec, return_labels=True)
        struct_pts, eng_pts = split_voxels(struct_grid, eng_grid)
        anat = denormalize_params(torch.from_numpy(pop[bi][:ANATOMICAL_DIM]))

        socketio.emit('generation', {
            'generation':    gen,
            'best_fitness':  float(fits[bi]),
            'avg_fitness':   float(np.mean(fits)),
            'worst_fitness': float(np.min(fits)),
            'breakdown':     brks[bi],
            'anatomy':       anat,
            'voxels':        struct_pts,
            'engine_voxels': eng_pts,
            'stagnation':    int(stagnation),
        })

        if stagnation >= 25:
            keep_n = POP_SIZE // 2
            order  = np.argsort(fits)[::-1]
            keep   = pop[order[:keep_n]]
            fresh  = evo.initial_population(user_spec)[:POP_SIZE - keep_n]
            pop    = np.concatenate([keep, fresh], axis=0)
            stagnation = 0
            print(f"[evo] stagnation restart at gen {gen}")
            continue

        pop = evo.select_and_reproduce(pop, fits, gen=gen,
                                       total_gens=generations,
                                       stagnation=stagnation)

    if best_ever is not None:
        struct_grid, eng_grid = evo.decode_voxels_hires(
            best_ever, user_spec=user_spec, return_labels=True)
        struct_pts, eng_pts = split_voxels(struct_grid, eng_grid)
        anat = denormalize_params(torch.from_numpy(best_ever[:ANATOMICAL_DIM]))
        _, brk = evo.evaluate(best_ever[None, :], user_spec)
        socketio.emit('generation', {
            'generation':    generations,
            'best_fitness':  float(best_fit),
            'avg_fitness':   float(best_fit),
            'worst_fitness': float(best_fit),
            'breakdown':     brk[0],
            'anatomy':       anat,
            'voxels':        struct_pts,
            'engine_voxels': eng_pts,
            'final':         True,
        })

    socketio.emit('done', {'best_fitness': best_fit})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start():
    global _thread, _stop
    if _thread is not None and _thread.is_alive():
        return jsonify({'error': 'Evolution already running'}), 400

    d = request.json or {}
    try:
        l = float(d['l']); h = float(d['h']); w = float(d['w'])
        user_spec = {
            'l': l, 'h': h, 'w': w,
            'fuse_l': float(d.get('fuse_l', l)),
            'fuse_h': float(d.get('fuse_h', h)),
            'fuse_w': float(d.get('fuse_w', min(h, w))),
            # NEW: hard engine envelope (m). 0 = no cap.
            'engine_l': float(d.get('engine_l', 0) or 0),
            'engine_h': float(d.get('engine_h', 0) or 0),
            'engine_w': float(d.get('engine_w', 0) or 0),
            'target_mass':      float(d['target_mass']),
            'engine_thrust_kN': float(d['engine_thrust_kN']),
            'areal_density':    float(d['areal_density']),
            'max_distance_km':  float(d['max_distance_km']),
            'cruise_speed':     float(d['cruise_speed']),
            'n_engines':        int(d.get('n_engines', 0)),
            'payload_mass':     float(d.get('payload_mass', 0)) or None,
        }
        gens = int(d.get('generations', DEFAULT_GENS))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'bad input: {e}'}), 400

    _stop = False
    _thread = threading.Thread(target=run_evolution,
                               args=(user_spec, gens), daemon=True)
    _thread.start()
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop():
    global _stop
    _stop = True
    return jsonify({'status': 'stopping'})


if __name__ == '__main__':
    MODEL = load_or_train_model()
    socketio.run(app, host='0.0.0.0', port=5011,
                 debug=False, allow_unsafe_werkzeug=True)