import numpy as np
import torch

from advae   import ANATOMICAL_DIM, LATENT_DIM, denormalize_params
from dataset import voxelize_jet, HIRES_GRID, PARAM_KEYS, PARAM_RANGES
from physics import evaluate_fitness, classify_vehicle, classify_tail, project_config


def _norm_of(key, value):
    lo, hi = PARAM_RANGES[key]
    return float(np.clip((value - lo) / (hi - lo) * 2 - 1, -1.0, 1.0))


def _denorm_of(key, norm_val):
    lo, hi = PARAM_RANGES[key]
    return float((norm_val + 1.0) / 2.0 * (hi - lo) + lo)


TAIL_CONFIGS = ['conventional', 't_tail', 'cruciform', 'v_tail', 'flying_wing']


def _is_strategic_lifter(user_spec):
    if user_spec is None: return False
    m = user_spec.get('target_mass', 0)
    if m < 120000: return False
    W = float(user_spec.get('w', 0) or 0)
    fuse_w = float(user_spec.get('fuse_w') or 0)
    fuse_l = float(user_spec.get('fuse_l') or 0)
    L = float(user_spec.get('l', 0) or 0)
    if W <= 0 or L <= 0: return False
    narrow = fuse_w > 0 and fuse_w / W < 0.35
    long_f = fuse_l > 0 and fuse_l / L > 0.70
    return narrow and long_f


class Evolution:
    def __init__(self, model, device='cpu', pop_size=120):
        self.model    = model.to(device).eval()
        self.device   = device
        self.pop_size = pop_size
        self.anat     = ANATOMICAL_DIM
        self.lat      = ANATOMICAL_DIM
        self._ix = {k: PARAM_KEYS.index(k) for k in PARAM_KEYS}

    def _set_tail(self, pop, i, cfg):
        ix = self._ix
        if cfg == 'flying_wing':
            pop[i, ix['has_vtail']]  = _norm_of('has_vtail', 0.0)
            pop[i, ix['has_htail']]  = _norm_of('has_htail', 0.0)
            pop[i, ix['vtail_cant']] = _norm_of('vtail_cant', 0.0)
            pop[i, ix['wing_sweep']] = _norm_of('wing_sweep', np.random.uniform(30, 55))
            pop[i, ix['wing_taper']] = _norm_of('wing_taper', np.random.uniform(0.15, 0.35))
        elif cfg == 'conventional':
            pop[i, ix['has_vtail']]  = _norm_of('has_vtail', 0.95)
            pop[i, ix['has_htail']]  = _norm_of('has_htail', 0.95)
            pop[i, ix['htail_z']]    = _norm_of('htail_z', np.random.uniform(-0.1, 0.2))
            pop[i, ix['vtail_cant']] = _norm_of('vtail_cant', np.random.uniform(0, 8))
        elif cfg == 't_tail':
            pop[i, ix['has_vtail']]  = _norm_of('has_vtail', 0.95)
            pop[i, ix['has_htail']]  = _norm_of('has_htail', 0.95)
            pop[i, ix['htail_z']]    = _norm_of('htail_z', np.random.uniform(0.90, 0.99))
            pop[i, ix['vtail_cant']] = _norm_of('vtail_cant', np.random.uniform(0, 5))
        elif cfg == 'cruciform':
            pop[i, ix['has_vtail']]  = _norm_of('has_vtail', 0.95)
            pop[i, ix['has_htail']]  = _norm_of('has_htail', 0.95)
            pop[i, ix['htail_z']]    = _norm_of('htail_z', np.random.uniform(0.42, 0.58))
            pop[i, ix['vtail_cant']] = _norm_of('vtail_cant', 0.0)
        elif cfg == 'v_tail':
            pop[i, ix['has_vtail']]  = _norm_of('has_vtail', 0.95)
            pop[i, ix['has_htail']]  = _norm_of('has_htail', 0.0)
            pop[i, ix['vtail_cant']] = _norm_of('vtail_cant', np.random.uniform(30, 48))
            pop[i, ix['vtail_size']] = _norm_of('vtail_size', np.random.uniform(0.12, 0.18))

    def _classify_individual(self, ind):
        ix = self._ix
        has_v = _denorm_of('has_vtail', ind[ix['has_vtail']]) > 0.5
        has_h = _denorm_of('has_htail', ind[ix['has_htail']]) > 0.5
        cant  = _denorm_of('vtail_cant', ind[ix['vtail_cant']])
        htz   = _denorm_of('htail_z', ind[ix['htail_z']])
        if not has_v and not has_h: return 'flying_wing'
        if has_v and not has_h and cant > 20: return 'v_tail'
        if has_v and has_h and htz >= 0.80:    return 't_tail'
        if has_v and has_h and 0.35 <= htz <= 0.65: return 'cruciform'
        if has_v and has_h: return 'conventional'
        return 'other'

    def _repair_tails(self, pop):
        ix = self._ix
        for i in range(pop.shape[0]):
            has_v_raw = _denorm_of('has_vtail', pop[i, ix['has_vtail']]) > 0.5
            has_h_raw = _denorm_of('has_htail', pop[i, ix['has_htail']]) > 0.5
            cant_deg  = _denorm_of('vtail_cant', pop[i, ix['vtail_cant']])
            htz       = _denorm_of('htail_z',    pop[i, ix['htail_z']])

            # h_only -> snap fin back on
            if has_h_raw and not has_v_raw:
                pop[i, ix['has_vtail']] = _norm_of('has_vtail', 0.95)
                has_v_raw = True

            # H-tail with canted V-tail and high htz -> stab would float, drop it
            if has_v_raw and has_h_raw and cant_deg > 20.0 and htz > 0.20:
                pop[i, ix['has_htail']] = _norm_of('has_htail', 0.0)
                has_h_raw = False

            # H-tail with canted fin -> force htz low
            if has_h_raw and has_v_raw and cant_deg > 20.0:
                pop[i, ix['htail_z']] = _norm_of('htail_z', min(htz, 0.10))

    def _seed_airlifter(self, pop):
        ix = self._ix
        probs = [('t_tail', 0.85), ('conventional', 0.10), ('cruciform', 0.05)]
        cfgs, ps = zip(*probs)
        for i in range(self.pop_size):
            pop[i, ix['n_engines_norm']]  = _norm_of('n_engines_norm', 4)
            pop[i, ix['engine_y_spread']] = _norm_of('engine_y_spread', np.random.uniform(0.30, 0.48))
            pop[i, ix['engine_x']]        = _norm_of('engine_x', np.random.uniform(0.42, 0.62))
            pop[i, ix['engine_length']]   = _norm_of('engine_length', np.random.uniform(0.14, 0.22))
            pop[i, ix['engine_size']]     = _norm_of('engine_size', np.random.uniform(0.14, 0.20))
            pop[i, ix['wing_span']]       = _norm_of('wing_span', np.random.uniform(0.82, 0.98))
            pop[i, ix['wing_root_chord']] = _norm_of('wing_root_chord', np.random.uniform(0.16, 0.26))
            pop[i, ix['wing_taper']]      = _norm_of('wing_taper', np.random.uniform(0.25, 0.45))
            pop[i, ix['wing_sweep']]      = _norm_of('wing_sweep', np.random.uniform(28, 38))
            pop[i, ix['wing_x']]          = _norm_of('wing_x', np.random.uniform(0.44, 0.58))
            pop[i, ix['wing_z']]          = _norm_of('wing_z', np.random.uniform(0.55, 0.88))
            pop[i, ix['wing_thickness']]  = _norm_of('wing_thickness', np.random.uniform(0.09, 0.13))
            pop[i, ix['wing_dihedral']]   = _norm_of('wing_dihedral', np.random.uniform(-2.5, 1.0))
            pop[i, ix['fuselage_length']] = _norm_of('fuselage_length', np.random.uniform(0.88, 0.98))
            pop[i, ix['fuselage_radius']] = _norm_of('fuselage_radius', np.random.uniform(0.85, 1.0))
            pop[i, ix['nose_fineness']]   = _norm_of('nose_fineness', np.random.uniform(0.10, 0.16))
            pop[i, ix['tail_fineness']]   = _norm_of('tail_fineness', np.random.uniform(0.24, 0.34))
            pop[i, ix['vtail_size']]      = _norm_of('vtail_size', np.random.uniform(0.11, 0.15))
            pop[i, ix['vtail_sweep']]     = _norm_of('vtail_sweep', np.random.uniform(35, 50))
            pop[i, ix['htail_span']]      = _norm_of('htail_span', np.random.uniform(0.22, 0.32))
            pop[i, ix['htail_chord']]     = _norm_of('htail_chord', np.random.uniform(0.08, 0.12))
            cfg = np.random.choice(cfgs, p=np.array(ps)/sum(ps))
            self._set_tail(pop, i, cfg)

    def _seed_balanced(self, pop, eng_choices=(1,2), is_airliner=False):
        """Seeds equally across all 5 tail topologies — guarantees every
        configuration gets explored, including flying-wing.
        For airliners we bias engines toward podded mounts and bodies toward
        a fat fuselage that fills the box."""
        ix = self._ix
        cfgs = TAIL_CONFIGS

        if is_airliner:
            ey_lo, ey_hi = 0.22, 0.45      # forces wing-podded twins/quads
            fr_lo, fr_hi = 0.75, 1.00      # nice fat tube
        else:
            ey_lo, ey_hi = 0.0, 0.40
            fr_lo, fr_hi = 0.30, 0.80

        for i in range(self.pop_size):
            pop[i, ix['engine_x']]        = _norm_of('engine_x', np.random.uniform(0.20, 0.85))
            pop[i, ix['engine_y_spread']] = _norm_of('engine_y_spread', np.random.uniform(ey_lo, ey_hi))
            pop[i, ix['engine_length']]   = _norm_of('engine_length', np.random.uniform(0.08, 0.22))
            pop[i, ix['engine_size']]     = _norm_of('engine_size', np.random.uniform(0.06, 0.18))
            pop[i, ix['wing_x']]          = _norm_of('wing_x', np.random.uniform(0.25, 0.75))
            pop[i, ix['wing_span']]       = _norm_of('wing_span', np.random.uniform(0.50, 0.95))
            pop[i, ix['wing_root_chord']] = _norm_of('wing_root_chord', np.random.uniform(0.10, 0.30))
            pop[i, ix['wing_taper']]      = _norm_of('wing_taper', np.random.uniform(0.15, 0.55))
            pop[i, ix['wing_sweep']]      = _norm_of('wing_sweep', np.random.uniform(0, 50))
            pop[i, ix['wing_z']]          = _norm_of('wing_z', np.random.uniform(-0.5, 0.5))
            pop[i, ix['wing_thickness']]  = _norm_of('wing_thickness', np.random.uniform(0.08, 0.14))
            pop[i, ix['wing_dihedral']]   = _norm_of('wing_dihedral', np.random.uniform(-1, 4))
            pop[i, ix['fuselage_length']] = _norm_of('fuselage_length', np.random.uniform(0.65, 0.97))
            pop[i, ix['fuselage_radius']] = _norm_of('fuselage_radius', np.random.uniform(fr_lo, fr_hi))
            pop[i, ix['nose_fineness']]   = _norm_of('nose_fineness', np.random.uniform(0.10, 0.25))
            pop[i, ix['tail_fineness']]   = _norm_of('tail_fineness', np.random.uniform(0.15, 0.35))
            pop[i, ix['vtail_size']]      = _norm_of('vtail_size', np.random.uniform(0.06, 0.16))
            pop[i, ix['vtail_sweep']]     = _norm_of('vtail_sweep', np.random.uniform(20, 55))
            pop[i, ix['htail_span']]      = _norm_of('htail_span', np.random.uniform(0.18, 0.35))
            pop[i, ix['htail_chord']]     = _norm_of('htail_chord', np.random.uniform(0.07, 0.13))
            pop[i, ix['n_engines_norm']]  = _norm_of('n_engines_norm', np.random.choice(eng_choices))

            # Round-robin across 5 topologies -> equal coverage
            cfg = cfgs[i % len(cfgs)]
            self._set_tail(pop, i, cfg)

    def initial_population(self, user_spec=None):
        cls = classify_vehicle(user_spec) if user_spec else 'drone'
        pop = np.random.uniform(-0.25, 0.25,
                                (self.pop_size, self.anat)).astype(np.float32)

        if cls == 'airliner' and _is_strategic_lifter(user_spec):
            print('[evo] strategic-airlifter seeding (high-wing, T-tail, 4-podded)')
            self._seed_airlifter(pop)
        elif cls == 'airliner':
            self._seed_balanced(pop, eng_choices=(2, 2, 3, 4), is_airliner=True)
        elif cls == 'jet':
            self._seed_balanced(pop, eng_choices=(1, 2, 2))
        else:
            self._seed_balanced(pop, eng_choices=(1, 1, 2))

        if user_spec is not None:
            n_user = int(user_spec.get('n_engines', 0) or 0)
            if n_user > 0:
                pop[:, self._ix['n_engines_norm']] = _norm_of('n_engines_norm',
                                                              np.clip(n_user, 1, 4))
        np.clip(pop, -1.0, 1.0, out=pop)
        self._repair_tails(pop)
        return pop

    def evaluate(self, pop, user_spec):
        fits, brks = [], []
        for ind in pop:
            anat = denormalize_params(torch.from_numpy(ind[:self.anat]))
            f, b = evaluate_fitness(anat, user_spec)
            fits.append(f); brks.append(b)
        return np.array(fits), brks

    def select_and_reproduce(self, pop, fits,
                             gen=0, total_gens=150, stagnation=0):
        order = np.argsort(fits)[::-1]
        n_fit_elite = max(2, self.pop_size // 20)
        fit_elites = [pop[i].copy() for i in order[:n_fit_elite]]

        # Tail-type diversity elites — best individual from EACH topology.
        seen = set()
        topo_elites = []
        for idx in order:
            t = self._classify_individual(pop[idx])
            if t not in seen:
                seen.add(t)
                topo_elites.append(pop[idx].copy())
                if len(seen) >= len(TAIL_CONFIGS):
                    break

        # Spatial diversity elites
        n_div_elite = max(2, self.pop_size // 25)
        remaining = list(order[n_fit_elite:])
        div_elites = []
        reference = list(fit_elites) + list(topo_elites)
        for _ in range(n_div_elite):
            if not remaining: break
            best_idx, best_d = None, -1.0
            for idx in remaining:
                d = min(np.linalg.norm(pop[idx] - r) for r in reference)
                if d > best_d:
                    best_d, best_idx = d, idx
            if best_idx is None: break
            div_elites.append(pop[best_idx].copy())
            reference.append(pop[best_idx])
            remaining.remove(best_idx)

        new_pop = fit_elites + topo_elites + div_elites

        progress = gen / max(total_gens, 1)
        base_std = 0.18 * (1.0 - progress) + 0.03
        mut_rate = 0.7
        cx_dims  = 0.5
        if stagnation > 15:
            base_std *= 2.5
            mut_rate  = 0.9
            cx_dims   = 0.35

        def tourn():
            a, b = np.random.choice(self.pop_size, 2, replace=False)
            return pop[a] if fits[a] > fits[b] else pop[b]

        while len(new_pop) < self.pop_size:
            p1, p2 = tourn(), tourn()
            mask   = np.random.rand(self.lat) < cx_dims
            child  = np.where(mask, p1, p2).astype(np.float32)
            if np.random.rand() < mut_rate:
                k = np.random.randint(1, 5)
                for _ in range(k):
                    d = np.random.randint(0, self.anat)
                    child[d] += np.random.normal(0, base_std)
            child[:self.anat] = np.clip(child[:self.anat], -1.0, 1.0)
            new_pop.append(child)

        out = np.array(new_pop[:self.pop_size], dtype=np.float32)
        self._repair_tails(out)
        return out

    def decode_voxels_vae(self, ind):
        with torch.no_grad():
            z = np.zeros(LATENT_DIM, dtype=np.float32)
            z[:min(self.lat, LATENT_DIM)] = ind[:min(self.lat, LATENT_DIM)]
            zt = torch.from_numpy(z).unsqueeze(0).to(self.device)
            return self.model.decode(zt).cpu().numpy()[0, 0]

    def decode_voxels_hires(self, ind, grid=HIRES_GRID, user_spec=None,
                            return_labels=False):
        anat = denormalize_params(torch.from_numpy(ind[:self.anat]))
        cls = classify_vehicle(user_spec) if user_spec else 'drone'
        anat = project_config(anat, cls, user_spec)
        if user_spec is not None:
            return voxelize_jet(
                anat, grid=grid,
                L=user_spec['l'], H=user_spec['h'], W=user_spec['w'],
                fuse_l=user_spec.get('fuse_l'),
                fuse_h=user_spec.get('fuse_h'),
                fuse_w=user_spec.get('fuse_w'),
                engine_l_cap=user_spec.get('engine_l') or None,
                engine_h_cap=user_spec.get('engine_h') or None,
                engine_w_cap=user_spec.get('engine_w') or None,
                return_labels=return_labels)
        return voxelize_jet(anat, grid=grid, return_labels=return_labels)

    decode_voxels            = decode_voxels_hires
    decode_voxels_analytical = decode_voxels_hires