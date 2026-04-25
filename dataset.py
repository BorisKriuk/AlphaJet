import numpy as np

GRID = 32
HIRES_GRID = 384

PARAM_KEYS = [
    'fuselage_length', 'fuselage_radius', 'nose_fineness', 'tail_fineness',
    'wing_span', 'wing_root_chord', 'wing_taper', 'wing_sweep',
    'wing_dihedral', 'wing_x', 'wing_z', 'wing_thickness',
    'has_vtail', 'vtail_size', 'vtail_sweep', 'vtail_cant',
    'has_htail', 'htail_span', 'htail_chord', 'htail_z',
    'n_engines_norm', 'engine_length', 'engine_size',
    'engine_x', 'engine_y_spread',
]

PARAM_RANGES = {
    'fuselage_length': (0.55, 0.98),
    'fuselage_radius': (0.30, 1.00),
    'nose_fineness':   (0.08, 0.35),
    'tail_fineness':   (0.12, 0.45),
    'wing_span':       (0.35, 0.98),
    'wing_root_chord': (0.08, 0.45),
    'wing_taper':      (0.10, 0.95),
    'wing_sweep':      (0.0, 65.0),
    'wing_dihedral':   (-3.0, 8.0),
    'wing_x':          (0.20, 0.78),
    'wing_z':          (-0.9, 0.9),
    'wing_thickness':  (0.06, 0.18),
    'has_vtail':       (0.0, 1.0),
    'vtail_size':      (0.04, 0.20),
    'vtail_sweep':     (10.0, 65.0),
    'vtail_cant':      (0.0, 55.0),
    'has_htail':       (0.0, 1.0),
    'htail_span':      (0.12, 0.55),
    'htail_chord':     (0.05, 0.18),
    'htail_z':         (-0.3, 1.0),
    'n_engines_norm':  (1.0, 4.0),
    'engine_length':   (0.06, 0.28),
    'engine_size':     (0.04, 0.22),
    'engine_x':        (0.10, 0.95),
    'engine_y_spread': (0.0, 0.55),
}

_DEFAULT_L, _DEFAULT_H, _DEFAULT_W = 38.0, 11.0, 34.0

# Bumped — fuselage now has nose-droop + tail-upsweep, single engine is rear-mounted
DATASET_VERSION = "v5-asym-fuse+rear-single-eng"

_TAIL_CONFIGS = ['conventional', 't_tail', 'cruciform', 'v_tail', 'flying_wing']
_TAIL_PROBS   = [0.22,            0.20,     0.13,        0.20,     0.25]


def _apply_tail_config(p, cfg, rng):
    if cfg == 'conventional':
        p['has_vtail']   = float(rng.uniform(0.65, 1.0))
        p['has_htail']   = float(rng.uniform(0.65, 1.0))
        p['htail_z']     = float(rng.uniform(-0.10, 0.20))
        p['vtail_cant']  = float(rng.uniform(0.0, 10.0))
    elif cfg == 't_tail':
        p['has_vtail']   = float(rng.uniform(0.65, 1.0))
        p['has_htail']   = float(rng.uniform(0.65, 1.0))
        p['htail_z']     = float(rng.uniform(0.88, 1.0))
        p['vtail_cant']  = float(rng.uniform(0.0, 6.0))
    elif cfg == 'cruciform':
        p['has_vtail']   = float(rng.uniform(0.65, 1.0))
        p['has_htail']   = float(rng.uniform(0.65, 1.0))
        p['htail_z']     = float(rng.uniform(0.40, 0.62))
        p['vtail_cant']  = float(rng.uniform(0.0, 5.0))
    elif cfg == 'v_tail':
        p['has_vtail']   = float(rng.uniform(0.65, 1.0))
        p['has_htail']   = float(rng.uniform(0.0, 0.35))
        p['vtail_cant']  = float(rng.uniform(28.0, 52.0))
        p['vtail_size']  = float(rng.uniform(0.11, 0.19))
    else:  # flying_wing
        p['has_vtail']   = float(rng.uniform(0.0, 0.35))
        p['has_htail']   = float(rng.uniform(0.0, 0.35))
        p['wing_sweep']  = float(rng.uniform(28.0, 60.0))
        p['wing_taper']  = float(rng.uniform(0.12, 0.40))
    return p


def sample_params(rng):
    p = {k: float(rng.uniform(*PARAM_RANGES[k])) for k in PARAM_KEYS}
    cfg = rng.choice(_TAIL_CONFIGS, p=_TAIL_PROBS)
    _apply_tail_config(p, cfg, rng)
    for k in PARAM_KEYS:
        lo, hi = PARAM_RANGES[k]
        p[k] = float(np.clip(p[k], lo, hi))
    return p


def voxelize_jet(params, grid=GRID, L=None, H=None, W=None,
                 fuse_l=None, fuse_h=None, fuse_w=None,
                 engine_l_cap=None, engine_h_cap=None, engine_w_cap=None,
                 return_labels=False):
    """+x=nose, +y=right, +z=up.

    Modern-aircraft fuselage shape:
      * NOSE  – blunter taper (cockpit profile) + slight downward droop
      * TAIL  – sharper taper + characteristic UPSWEEP (cargo-style tail cone)
    Single engine is forced rear-mounted with the nozzle protruding behind
    the tail so it is always visible in the render.
    """
    if L is None: L = _DEFAULT_L
    if H is None: H = _DEFAULT_H
    if W is None: W = _DEFAULT_W
    if fuse_l is None: fuse_l = L
    if fuse_h is None: fuse_h = H
    if fuse_w is None: fuse_w = min(H, W)

    G = grid
    out = np.zeros((G, G, G), dtype=np.float32)
    eng_mask = np.zeros((G, G, G), dtype=np.float32) if return_labels else None

    M = max(L, H, W)
    vpm = (G - 2) / M
    one_vox = 1.0 / vpm

    I, J, K = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing='ij')
    x = (I - G/2.0) / vpm
    y = (J - G/2.0) / vpm
    z = (K - G/2.0) / vpm

    # ---------- FUSELAGE (asymmetric nose/tail) ----------
    fl = params['fuselage_length'] * fuse_l
    ry_max = params['fuselage_radius'] * fuse_w / 2.0
    rz_max = params['fuselage_radius'] * fuse_h / 2.0
    fr_y = max(one_vox * 1.2, ry_max)
    fr_z = max(one_vox * 1.2, rz_max)
    fr   = max(fr_y, fr_z)

    nose_len = max(one_vox * 1.5, params['nose_fineness'] * fl)
    tail_len = max(one_vox * 1.5, params['tail_fineness'] * fl)

    x_nose = fl / 2.0
    x_tail = -fl / 2.0
    x_nose_base = x_nose - nose_len
    x_tail_base = x_tail + tail_len

    scale_y  = np.ones_like(x)
    scale_z  = np.ones_like(x)
    z_center = np.zeros_like(x)

    # NOSE: blunt cockpit profile + small droop (modern airliner / fighter look)
    in_nose  = (x > x_nose_base) & (x <= x_nose)
    nose_t   = np.clip((x_nose - x) / nose_len, 0, 1)
    nose_sy  = nose_t ** 0.55
    nose_sz  = nose_t ** 0.48          # slightly blunter top -> cockpit hump
    scale_y  = np.where(in_nose, nose_sy, scale_y)
    scale_z  = np.where(in_nose, nose_sz, scale_z)
    nose_droop = -(1.0 - nose_t) * fr_z * 0.14   # nose tip below cabin CL
    z_center   = np.where(in_nose, nose_droop, z_center)

    # TAIL: sharper taper + upsweep (every transport/airliner has this)
    in_tail  = (x >= x_tail) & (x < x_tail_base)
    tail_t   = np.clip((x - x_tail) / tail_len, 0, 1)
    tail_sy  = tail_t ** 0.62
    tail_sz  = tail_t ** 0.78          # tail height collapses faster than width
    scale_y  = np.where(in_tail, tail_sy, scale_y)
    scale_z  = np.where(in_tail, tail_sz, scale_z)
    upsweep  = (1.0 - tail_t) * fr_z * 0.55      # tail cone rides UP, belly curves up
    z_center = np.where(in_tail, upsweep, z_center)

    local_ry = np.maximum(fr_y * scale_y, 1e-6)
    local_rz = np.maximum(fr_z * scale_z, 1e-6)
    fuselage = ((x >= x_tail) & (x <= x_nose) &
                ((y / local_ry)**2 + ((z - z_center) / local_rz)**2 < 1.0))
    out[fuselage] = 1.0

    # ---------- MAIN WING (yehudi break + raked tip + outboard t/c thinning) ----------
    ws         = params['wing_span'] * W / 2.0
    wc_root    = params['wing_root_chord'] * L
    wtap       = params['wing_taper']
    sweep_t    = np.tan(np.deg2rad(params['wing_sweep']))
    dih_t      = np.tan(np.deg2rad(params['wing_dihedral']))
    wing_x_LE  = (0.5 - params['wing_x']) * fl
    wing_z_abs = params['wing_z'] * fr_z
    tc         = params['wing_thickness']

    abs_y   = np.abs(y)
    y_frac  = np.clip(abs_y / max(ws, 1e-6), 0, 1)

    KINK_Y         = 0.32
    kink_chord_rat = 0.58 + 0.32 * wtap
    t_in  = np.clip(y_frac / KINK_Y, 0, 1)
    t_out = np.clip((y_frac - KINK_Y) / max(1.0 - KINK_Y, 1e-6), 0, 1)
    wc_inboard  = wc_root * (1.0 - (1.0 - kink_chord_rat) * t_in)
    wc_outboard = wc_root * (kink_chord_rat - (kink_chord_rat - wtap) * t_out)
    wc_loc = np.where(y_frac <= KINK_Y, wc_inboard, wc_outboard)

    INB_SWEEP_FRAC = 0.55
    RAKE_START     = 0.90
    RAKE_FACTOR    = 0.85
    inb_off  = sweep_t * INB_SWEEP_FRAC * abs_y
    out_off  = (sweep_t * INB_SWEEP_FRAC * (KINK_Y * ws) +
                sweep_t * (abs_y - KINK_Y * ws))
    xLE_loc  = wing_x_LE - np.where(y_frac <= KINK_Y, inb_off, out_off)
    rake_off = sweep_t * RAKE_FACTOR * np.maximum(abs_y - RAKE_START * ws, 0.0)
    xLE_loc -= rake_off

    TIP_CURL = 0.94
    curl_t   = np.clip((y_frac - TIP_CURL) / max(1.0 - TIP_CURL, 1e-6), 0, 1)
    z_curl   = 0.35 * tc * wc_root * curl_t**2

    xc_mid  = xLE_loc - wc_loc / 2.0
    z_cen   = wing_z_abs + abs_y * dih_t + z_curl

    tc_local = tc * (1.0 - 0.30 * y_frac)
    half_t   = np.maximum(wc_loc * tc_local * 0.5, one_vox * 0.6)

    chord_f = (x - xc_mid) / np.maximum(wc_loc, 1e-6)
    airfoil = half_t * np.sqrt(np.maximum(0, 1 - 4 * chord_f ** 2))
    wing = ((abs_y < ws) & (np.abs(chord_f) < 0.5) &
            (np.abs(z - z_cen) < airfoil))
    out[wing] = 1.0

    # ---------- V-TAIL / FIN ----------
    has_v = params['has_vtail'] > 0.5
    cant_deg = params.get('vtail_cant', 0.0)
    cant_r = np.deg2rad(cant_deg)
    vt_h_used = 0.0

    if has_v:
        vt_h_raw = params['vtail_size'] * L * 1.2
        vt_h_cap = max(min(H - fr_z, 0.22 * L), fr_z)
        vt_h     = min(vt_h_raw, vt_h_cap)
        vt_h_used = vt_h
        vt_chord_r = min(params['vtail_size'] * L * 1.1, 1.4 * vt_h)
        vt_sweep_t = np.tan(np.deg2rad(params['vtail_sweep']))
        vt_x_LE_r  = x_tail + vt_chord_r * 1.05
        vt_thick_r = max(one_vox * 1.0, min(L, H, W) / 60.0)

        if cant_deg < 3.0:
            z_pos  = np.maximum(z - fr_z * 0.4, 0.0)
            z_frac = np.clip(z_pos / max(vt_h, 1e-6), 0, 1)
            vt_chord_l = vt_chord_r * (1 - 0.55 * z_frac)
            vt_xLE_loc = vt_x_LE_r - vt_sweep_t * z_pos
            vt_thick_l = vt_thick_r * (1 - 0.5 * z_frac)
            vt = ((z > fr_z * 0.3) & (z < fr_z * 0.3 + vt_h) &
                  (np.abs(y) < vt_thick_l) &
                  (x < vt_xLE_loc) & (x > vt_xLE_loc - vt_chord_l))
            out[vt] = 1.0
        else:
            s, c = np.sin(cant_r), np.cos(cant_r)
            for side in (-1.0, 1.0):
                span_pos = side * y * s + z * c
                lat_pos  = side * y * c - z * s
                span_pos_pos = np.maximum(span_pos, 0.0)
                span_frac = np.clip(span_pos / max(vt_h, 1e-6), 0, 1)
                vt_chord_l = vt_chord_r * (1 - 0.55 * span_frac)
                vt_xLE_loc = vt_x_LE_r - vt_sweep_t * span_pos_pos
                vt_thick_l = vt_thick_r * (1 - 0.5 * span_frac)
                vt = ((span_pos > 0) & (span_pos < vt_h) &
                      (np.abs(lat_pos) < vt_thick_l) &
                      (x < vt_xLE_loc) & (x > vt_xLE_loc - vt_chord_l))
                out[vt] = 1.0

    # ---------- H-TAIL ----------
    if params['has_htail'] > 0.5:
        wing_half  = params['wing_span'] * W / 2.0
        hs         = params['htail_span'] * wing_half
        ht_chord_r = min(params['htail_chord'] * L, hs * 0.55)

        fin_is_vertical = has_v and cant_deg < 20.0 and vt_h_used > 0.0
        if fin_is_vertical:
            ref_h = vt_h_used
            htz   = params['htail_z']
        else:
            ref_h = fr_z
            htz   = min(params['htail_z'], 0.15)

        ht_z_abs   = fr_z * 0.3 + htz * ref_h
        ht_x_LE_r  = x_tail + ht_chord_r
        ht_sweep_t = 0.55
        ht_thick_r = max(one_vox * 0.9, min(L, H, W) / 70.0)
        y_frac_h   = np.clip(abs_y / max(hs, 1e-6), 0, 1)
        ht_chord_l = ht_chord_r * (1 - 0.55 * y_frac_h)
        ht_xLE_loc = ht_x_LE_r - ht_sweep_t * abs_y
        ht_thick_l = ht_thick_r * (1 - 0.5 * y_frac_h)
        ht = ((abs_y < hs) &
              (x < ht_xLE_loc) & (x > ht_xLE_loc - ht_chord_l) &
              (np.abs(z - ht_z_abs) < ht_thick_l))
        out[ht] = 1.0

    # ---------- ENGINES (HARD-CAPPED, single engine forced rear-mounted) ----------
    n_eng = int(np.clip(round(params['n_engines_norm']), 1, 4))
    eL_m  = max(one_vox * 1.2, params['engine_length'] * L)
    eS_m  = max(one_vox * 1.2, params['engine_size']   * min(H, W))
    eW_m  = eS_m
    eH_m  = eS_m

    if engine_l_cap is not None and engine_l_cap > 0: eL_m = min(eL_m, float(engine_l_cap))
    if engine_w_cap is not None and engine_w_cap > 0: eW_m = min(eW_m, float(engine_w_cap))
    if engine_h_cap is not None and engine_h_cap > 0: eH_m = min(eH_m, float(engine_h_cap))

    eX_m_user = (0.5 - params['engine_x']) * fl
    eY_m      = params['engine_y_spread'] * W / 2.0
    is_fuse   = eY_m < (W / 32.0)

    if n_eng == 1:
        # Single engine -> ALWAYS body-integrated and REAR-mounted, with the
        # nozzle protruding behind the tail cone (Geran-3 pusher / F-16 nozzle /
        # business-jet tail mount). Forced visible in render.
        is_fuse = True
        ex_one  = x_tail + eL_m * 0.30   # ~70% of length pokes out behind tail
        ez_one  = fr_z * 0.35            # sits in the upswept tail
        centers = [(ex_one, 0.0, ez_one)]
    elif is_fuse:
        eX_m   = eX_m_user
        offs_y = fr_y + eW_m * 0.55
        offs_z = fr_z + eH_m * 0.55
        if n_eng == 2:
            centers = [(eX_m, -offs_y, 0.0), (eX_m, offs_y, 0.0)]
        elif n_eng == 3:
            centers = [(eX_m, 0.0, offs_z),
                       (eX_m, -offs_y, 0.0), (eX_m, offs_y, 0.0)]
        else:
            centers = [(eX_m, -offs_y, 0.0), (eX_m, offs_y, 0.0),
                       (eX_m, 0.0, offs_z), (eX_m, 0.0, -offs_z)]
    else:
        eX_m   = eX_m_user
        ez_pod = wing_z_abs - eH_m * 0.8
        if n_eng == 2:
            centers = [(eX_m, -eY_m, ez_pod), (eX_m, eY_m, ez_pod)]
        elif n_eng == 3:
            centers = [(eX_m + eL_m*0.2, 0.0, 0.0),
                       (eX_m, -eY_m, ez_pod), (eX_m, eY_m, ez_pod)]
        else:
            centers = [(eX_m, -eY_m,      ez_pod), (eX_m, eY_m,      ez_pod),
                       (eX_m, -eY_m*0.55, ez_pod), (eX_m, eY_m*0.55, ez_pod)]

    for ex, ey, ez in centers:
        mask = ((np.abs(x - ex) < eL_m/2) &
                (np.abs(y - ey) < eW_m/2) &
                (np.abs(z - ez) < eH_m/2))
        out[mask] = 1.0
        if eng_mask is not None:
            eng_mask[mask] = 1.0
        if (n_eng > 1) and (not is_fuse) and abs(ez - wing_z_abs) > eH_m * 0.3:
            pylon_t = max(one_vox * 0.9, min(L, H, W) / 80.0)
            pyl = ((np.abs(y - ey) < pylon_t) &
                   (np.abs(x - ex) < eL_m * 0.25) &
                   (z > min(ez, wing_z_abs)) & (z < max(ez, wing_z_abs)))
            out[pyl] = 1.0

    if return_labels:
        return out, eng_mask
    return out