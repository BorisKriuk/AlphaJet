import numpy as np
from dataset import PARAM_RANGES

G_ACCEL       = 9.81
RHO_SL        = 1.225
RHO_CRUISE    = 0.40
A_SOUND       = 295.0

CL_MAX        = 1.5
YIELD_MPA     = 280.0
LIMIT_N       = 3.5
ULT_N         = 1.5 * LIMIT_N

KORN_KAPPA    = 0.87
WINGBOX_EFF   = 0.15
SYSTEMS_FRAC  = 0.08
T_MARGIN_LO   = 1.15
T_MARGIN_HI   = 1.50

MOUNT_MIN_PEN_FRAC  = 0.15
MOUNT_GOOD_PEN_FRAC = 0.35
MOUNT_MAX_GAP_FRAC  = 0.50

HT_MOUNT_MIN_PEN_FRAC  = 0.25
HT_MOUNT_GOOD_PEN_FRAC = 1.00
HT_MOUNT_MAX_GAP_FRAC  = 1.50

VV_MIN, VV_GOOD_LO, VV_GOOD_HI, VV_MAX = 0.015, 0.03, 0.09, 0.14
STATIC_MARGIN_LO = 0.05
STATIC_MARGIN_HI = 0.25

RHO_FUEL          = 810.0
RHO_PAYLOAD       = 1400.0
FUEL_VOL_FRAC_MAX = 0.75
VOL_PACK_EFF      = 0.85

STRUCT_MIN_FRAC   = 0.15
MIN_FUSE_LEN_FRAC = 0.50
CG_PLACE_TOL_MAC  = 0.60

SV_OVER_SREF_MAX  = 0.12

CHORD_RATIO_GOOD   = 0.28
CHORD_RATIO_HARD   = 0.55
CL_CRUISE_GOOD_LO  = 0.30
CL_CRUISE_GOOD_HI  = 0.60
CL_CRUISE_HARD_LO  = 0.15
SPAN_TO_L_GOOD_LO  = 0.60
SPAN_TO_L_GOOD_HI  = 1.50
DENSITY_REF_KG_M3  = 40.0
SHAPE_RATIO_HARD   = 4.0

WING_LOAD_GOOD = {
    'airliner': (350.0, 700.0),
    'jet':      (220.0, 550.0),
    'drone':    (90.0,  320.0),
}
WING_LOAD_HARD_LO = {'airliner': 200.0, 'jet': 130.0, 'drone': 50.0}


def classify_vehicle(user_spec):
    m = user_spec['target_mass']
    if m >= 30000: return 'airliner'
    if m >= 1000:  return 'jet'
    return 'drone'


def class_params(cls):
    if cls == 'airliner':
        return dict(tsfc=1.55e-5, ff_max=0.45, ld_target=19.0,
                    min_eng=2, max_eng=4, ar_target=9.0,
                    prefer_tube_and_wing=True)
    if cls == 'jet':
        return dict(tsfc=2.2e-5, ff_max=0.50, ld_target=15.0,
                    min_eng=1, max_eng=4, ar_target=7.0,
                    prefer_tube_and_wing=False)
    return     dict(tsfc=3.5e-5, ff_max=0.55, ld_target=11.0,
                    min_eng=1, max_eng=2, ar_target=6.0,
                    prefer_tube_and_wing=False)


def project_config(params, cls, user_spec=None):
    p = dict(params)
    if user_spec is None:
        return p

    n_user = int(user_spec.get('n_engines', 0) or 0)
    if n_user > 0:
        p['n_engines_norm'] = float(np.clip(n_user, 1, 4))

    L = float(user_spec.get('l', 0) or 0)
    H = float(user_spec.get('h', 0) or 0)
    W = float(user_spec.get('w', 0) or 0)

    eng_l_max = float(user_spec.get('engine_l', 0) or 0)
    if eng_l_max > 0 and L > 0:
        cap_norm = eng_l_max / L
        lo, hi = PARAM_RANGES['engine_length']
        p['engine_length'] = float(np.clip(p['engine_length'], lo, min(hi, cap_norm)))

    eng_h_max = float(user_spec.get('engine_h', 0) or 0)
    eng_w_max = float(user_spec.get('engine_w', 0) or 0)
    if eng_h_max > 0 and eng_w_max > 0 and min(H, W) > 0:
        cap_dim = min(eng_h_max, eng_w_max)
        cap_norm = cap_dim / min(H, W)
        lo, hi = PARAM_RANGES['engine_size']
        p['engine_size'] = float(np.clip(p['engine_size'], lo, min(hi, cap_norm)))

    return p


def classify_tail(params):
    has_v = params['has_vtail'] > 0.5
    has_h = params['has_htail'] > 0.5
    cant  = params.get('vtail_cant', 0.0)
    hz    = params.get('htail_z', 0.0)
    if not has_v and not has_h:           return 'flying_wing'
    if has_v and not has_h and cant > 20: return 'v_tail'
    if has_v and has_h and hz >= 0.80:    return 't_tail'
    if has_v and has_h and 0.35 <= hz <= 0.65: return 'cruciform'
    if has_v and has_h:                   return 'conventional'
    if has_v and not has_h:               return 'v_only'
    return 'h_only'


def _fuselage_radius_at(x, fl, fr_eq, nose_len, tail_len):
    x_nose = fl / 2.0; x_tail = -fl / 2.0
    if x > x_nose or x < x_tail: return 0.0
    x_nose_base = x_nose - nose_len
    x_tail_base = x_tail + tail_len
    if x > x_nose_base:
        t = np.clip((x_nose - x) / max(nose_len, 1e-6), 0, 1)
        # blunter cockpit profile (matches dataset.voxelize_jet)
        return fr_eq * (t**0.52)
    if x < x_tail_base:
        t = np.clip((x - x_tail) / max(tail_len, 1e-6), 0, 1)
        # sharper tail taper (matches dataset.voxelize_jet)
        return fr_eq * (t**0.70)
    return fr_eq


def _engine_centers_SI(params, fl, fr_y, fr_z, W_box, H_box, eS_m, eL_m,
                       wing_z_abs, n_eng):
    """Returns (centers, is_fuse). For n_eng==1, ALWAYS rear-mounted to keep
    the engine visible and well-attached to the rear bulkhead."""
    eX_m_user = (0.5 - params['engine_x']) * fl
    eY_m      = params['engine_y_spread'] * W_box / 2.0
    is_fuse   = eY_m < (W_box / 32.0)

    # ---- single engine: forced rear-mount (matches dataset.voxelize_jet) ----
    if n_eng == 1:
        x_tail = -fl / 2.0
        ex_one = x_tail + eL_m * 0.30
        ez_one = fr_z * 0.35
        return [(ex_one, 0.0, ez_one)], True

    offs_y = fr_y + eS_m * 0.55
    offs_z = fr_z + eS_m * 0.55
    if is_fuse:
        if n_eng == 2:
            centers = [(eX_m_user, -offs_y, 0.0), (eX_m_user, offs_y, 0.0)]
        elif n_eng == 3:
            centers = [(eX_m_user, 0.0, offs_z),
                       (eX_m_user, -offs_y, 0.0), (eX_m_user, offs_y, 0.0)]
        else:
            centers = [(eX_m_user, -offs_y, 0.0), (eX_m_user, offs_y, 0.0),
                       (eX_m_user, 0.0, offs_z), (eX_m_user, 0.0, -offs_z)]
    else:
        ez_pod = wing_z_abs - eS_m * 0.8
        if n_eng == 2:
            centers = [(eX_m_user, -eY_m, ez_pod), (eX_m_user, eY_m, ez_pod)]
        elif n_eng == 3:
            centers = [(eX_m_user + eL_m*0.2, 0.0, 0.0),
                       (eX_m_user, -eY_m, ez_pod), (eX_m_user, eY_m, ez_pod)]
        else:
            centers = [(eX_m_user, -eY_m,      ez_pod), (eX_m_user, eY_m,      ez_pod),
                       (eX_m_user, -eY_m*0.55, ez_pod), (eX_m_user, eY_m*0.55, ez_pod)]
    return centers, is_fuse


def _engine_mount_penetration(ex, ey, ez, eS_m, eL_m, fl, fr_eq, nose_len, tail_len,
                               b, cR, taper, sweep, dihedral_rad,
                               wing_x_param, wing_z_abs, tc):
    xs_probe = np.linspace(ex - eL_m/2.0, ex + eL_m/2.0, 9)

    r_locs = [_fuselage_radius_at(np.clip(x, -fl/2.0, fl/2.0),
                                   fl, fr_eq, nose_len, tail_len) for x in xs_probe]
    r_loc_max = max(r_locs) if r_locs else 0.0
    if r_loc_max <= 0.0:
        pen_fuse = -1e9
    else:
        r_center = np.hypot(ey, ez)
        pen_fuse = r_loc_max - (r_center - eS_m/2.0)

    y_abs = abs(ey)
    if y_abs > b/2.0 + 1e-6:
        pen_wing = -1e9
    else:
        y_frac   = y_abs / max(b/2.0, 1e-6)
        chord_y  = cR * (1.0 - (1.0 - taper) * y_frac)
        xLE_y    = (0.5 - wing_x_param)*fl - np.tan(sweep) * y_abs
        xTE_y    = xLE_y - chord_y
        z_wing_y = wing_z_abs + y_abs * np.tan(dihedral_rad)
        t_wing_y = chord_y * tc

        dxs = []
        for x in xs_probe:
            if   x > xLE_y: dxs.append(x - xLE_y)
            elif x < xTE_y: dxs.append(xTE_y - x)
            else:           dxs.append(0.0)
        dx_gap = min(dxs)

        z_wing_bot = z_wing_y - t_wing_y / 2.0
        z_eng_top  = ez + eS_m / 2.0
        pylon_gap  = z_wing_bot - z_eng_top

        if dx_gap > 0.0:
            pen_wing = -np.hypot(dx_gap, max(-pylon_gap, 0.0))
        else:
            ideal = 0.18 * eS_m
            tol   = 0.55 * eS_m
            if pylon_gap < -0.10 * eS_m:
                pen_wing = pylon_gap
            elif pylon_gap < 0.0:
                pen_wing = 0.30 * eS_m * (1.0 + pylon_gap / (0.10 * eS_m))
            elif pylon_gap <= tol:
                pen_wing = 0.50 * eS_m - abs(pylon_gap - ideal)
            else:
                pen_wing = -(pylon_gap - tol)

    return max(pen_fuse, pen_wing)


def _mount_score_from_pen(pen, eS_m):
    min_pen  = MOUNT_MIN_PEN_FRAC  * eS_m
    good_pen = MOUNT_GOOD_PEN_FRAC * eS_m
    max_gap  = MOUNT_MAX_GAP_FRAC  * eS_m
    if pen >= good_pen:
        return 1.0
    if pen >= min_pen:
        return 0.75 + 0.25 * (pen - min_pen) / max(good_pen - min_pen, 1e-6)
    if pen >= 0.0:
        return 0.30 + 0.45 * pen / max(min_pen, 1e-6)
    if pen >= -max_gap:
        return max(0.0, 0.30 * (1.0 + pen / max_gap))
    return 0.0


def _engine_attachment_score(params, fl, fr_y, fr_z, fr_eq, nose_len, tail_len,
                             b, cR, taper, sweep, dihedral_rad,
                             wing_z_abs, tc, W_box, H_box,
                             eS_m, eL_m, n_eng):
    centers, is_fuse = _engine_centers_SI(params, fl, fr_y, fr_z, W_box, H_box,
                                          eS_m, eL_m, wing_z_abs, n_eng)
    # Single engine is rear-bulkhead-mounted by construction -> firm.
    if n_eng == 1:
        return 1.0, MOUNT_GOOD_PEN_FRAC * eS_m, True, centers
    scores, pens = [], []
    for (ex, ey, ez) in centers:
        pen = _engine_mount_penetration(ex, ey, ez, eS_m, eL_m, fl, fr_eq,
                                        nose_len, tail_len,
                                        b, cR, taper, sweep, dihedral_rad,
                                        params['wing_x'], wing_z_abs, tc)
        pens.append(pen)
        scores.append(_mount_score_from_pen(pen, eS_m))
    return float(np.min(scores)), float(np.min(pens)), is_fuse, centers


def _ground_clearance_score(centers, is_fuse, eS_m, fr_z):
    if is_fuse: return 1.0
    belly_z = -fr_z;  allowed_drop = 1.5 * fr_z;  worst = 1.0
    for (_, _, ez) in centers:
        drop = belly_z - (ez - eS_m/2.0)
        s = 1.0 if drop <= 0 else max(0.0, 1.0 - drop / allowed_drop)
        worst = min(worst, s)
    return float(worst)


def _htail_attachment_pen(fl, fr_eq, fr_z, nose_len, tail_len,
                          ht_x_LE_r, ht_chord_r, ht_thick_r, ht_z_abs,
                          has_v, cant_deg, vt_h_eff, vt_chord_r, vt_sweep_t,
                          x_tail):
    xs = np.linspace(ht_x_LE_r - ht_chord_r, ht_x_LE_r, 7)
    pen_fuse = -1e9
    for xv in xs:
        xc = np.clip(xv, -fl/2.0, fl/2.0)
        r_loc = _fuselage_radius_at(xc, fl, fr_eq, nose_len, tail_len)
        if r_loc > 0.0:
            p = (r_loc + ht_thick_r) - abs(ht_z_abs)
            pen_fuse = max(pen_fuse, p)

    pen_fin = -1e9
    if has_v and cant_deg < 20.0 and vt_h_eff > 0.0:
        z_fin_base = fr_z * 0.3
        z_fin_top  = z_fin_base + vt_h_eff
        if z_fin_base <= ht_z_abs <= z_fin_top:
            z_gap = 0.0
        else:
            z_gap = max(z_fin_base - ht_z_abs, ht_z_abs - z_fin_top)

        z_in   = float(np.clip(ht_z_abs - z_fin_base, 0.0, vt_h_eff))
        z_frac = z_in / max(vt_h_eff, 1e-6)
        vt_c_loc   = vt_chord_r * (1.0 - 0.55 * z_frac)
        vt_xLE_loc = (x_tail + vt_chord_r * 1.05) - vt_sweep_t * z_in
        vt_xTE_loc = vt_xLE_loc - vt_c_loc

        x_overlap = min(ht_x_LE_r, vt_xLE_loc) - max(ht_x_LE_r - ht_chord_r, vt_xTE_loc)

        if z_gap <= 0.0 and x_overlap > 0.0:
            pen_fin = x_overlap
        elif x_overlap > 0.0:
            pen_fin = -z_gap
        else:
            pen_fin = -float(np.hypot(max(z_gap, 0.0), max(-x_overlap, 0.0)))

    return max(pen_fuse, pen_fin)


def _htail_mount_score(pen, ht_thick_r):
    min_pen  = HT_MOUNT_MIN_PEN_FRAC  * ht_thick_r
    good_pen = HT_MOUNT_GOOD_PEN_FRAC * ht_thick_r
    max_gap  = HT_MOUNT_MAX_GAP_FRAC  * ht_thick_r
    if pen >= good_pen: return 1.0
    if pen >= min_pen:  return 0.75 + 0.25 * (pen - min_pen) / max(good_pen - min_pen, 1e-6)
    if pen >= 0.0:      return 0.30 + 0.45 * pen / max(min_pen, 1e-6)
    if pen >= -max_gap: return max(0.0, 0.30 * (1.0 + pen / max_gap))
    return 0.0


def _cg_x(masses, xs):
    return float(np.sum(masses*xs) / max(np.sum(masses), 1e-6))


def _static_margin_score(sm):
    if STATIC_MARGIN_LO <= sm <= STATIC_MARGIN_HI: return 1.0
    if sm > 0: return max(0.0, 1.0 - abs(sm - 0.15) / 0.25)
    return max(0.0, 0.4 + 2.0*sm)


def _vtail_volume_score(Vv):
    if Vv <= 0.0:        return 0.0
    if Vv < VV_MIN:      return 0.2 * Vv / VV_MIN
    if Vv < VV_GOOD_LO:  return 0.2 + 0.8*(Vv - VV_MIN)/(VV_GOOD_LO - VV_MIN)
    if Vv <= VV_GOOD_HI: return 1.0
    if Vv < VV_MAX:      return 1.0 - 0.6*(Vv - VV_GOOD_HI)/(VV_MAX - VV_GOOD_HI)
    return 0.3


def _chord_realism(chord_ratio):
    if chord_ratio <= CHORD_RATIO_GOOD:
        return 1.0
    if chord_ratio <= 0.45:
        return float(max(0.30, 1.0 - 1.4 * (chord_ratio - CHORD_RATIO_GOOD)))
    return float(max(0.05, 0.65 - 1.6 * (chord_ratio - 0.45)))


def _cl_realism(CLc):
    if CL_CRUISE_GOOD_LO <= CLc <= CL_CRUISE_GOOD_HI:
        return 1.0
    if CLc < CL_CRUISE_GOOD_LO:
        return float(max(0.05, (CLc / CL_CRUISE_GOOD_LO) ** 2))
    return float(max(0.30, 1.0 - 0.8 * (CLc - CL_CRUISE_GOOD_HI)))


def _span_realism(bL):
    if SPAN_TO_L_GOOD_LO <= bL <= SPAN_TO_L_GOOD_HI:
        return 1.0
    if bL < SPAN_TO_L_GOOD_LO:
        return float(max(0.40, bL / SPAN_TO_L_GOOD_LO))
    return float(max(0.15, 1.0 - 0.55 * (bL - SPAN_TO_L_GOOD_HI)))


def _block_economy(shape_ratio):
    if shape_ratio <= 1.0:
        return 1.0
    if shape_ratio <= 2.5:
        return float(1.0 - 0.50 * (shape_ratio - 1.0) / 1.5)
    if shape_ratio <= 5.0:
        return float(max(0.15, 0.50 - 0.12 * (shape_ratio - 2.5)))
    return float(max(0.03, 0.20 - 0.04 * (shape_ratio - 5.0)))


def _wing_loading_realism(WoS, cls):
    lo, hi = WING_LOAD_GOOD.get(cls, (220.0, 550.0))
    if lo <= WoS <= hi:                    return 1.0
    if WoS < lo:                           return float(max(0.05, (WoS / lo) ** 2))
    return float(max(0.30, 1.0 - 0.6 * (WoS - hi) / hi))


def evaluate_fitness(params, user_spec):
    L = user_spec['l']; H = user_spec['h']; W = user_spec['w']
    fuse_L = float(user_spec.get('fuse_l') or L)
    fuse_H = float(user_spec.get('fuse_h') or H)
    fuse_W = float(user_spec.get('fuse_w') or min(H, W))
    fuse_L = min(fuse_L, L); fuse_H = min(fuse_H, H); fuse_W = min(fuse_W, W)

    m_target   = user_spec['target_mass']
    T_total    = user_spec['engine_thrust_kN'] * 1000.0
    areal_rho  = user_spec['areal_density']
    R_req      = user_spec['max_distance_km'] * 1000.0
    V          = user_spec['cruise_speed']

    cls = classify_vehicle(user_spec)
    cp  = class_params(cls)
    params = project_config(params, cls, user_spec)

    TSFC      = cp['tsfc']
    FF_MAX    = cp['ff_max']
    LD_TARGET = cp['ld_target']
    AR_TARGET = cp['ar_target']

    fl   = params['fuselage_length'] * fuse_L
    fr_y = params['fuselage_radius'] * fuse_W / 2.0
    fr_z = params['fuselage_radius'] * fuse_H / 2.0
    fr_eq = float(np.sqrt(max(fr_y, 1e-6) * max(fr_z, 1e-6)))
    fr   = max(fr_y, fr_z)
    fd   = 2.0 * fr_eq
    nose_len = params['nose_fineness'] * fl
    tail_len = params['tail_fineness'] * fl
    x_tail = -fl / 2.0

    b     = params['wing_span'] * W
    cR    = params['wing_root_chord'] * L
    taper = params['wing_taper']
    cT    = cR * taper
    Sref  = 0.5 * (cR + cT) * b
    AR    = b*b / max(Sref, 1e-6)
    MAC   = (2.0/3.0) * cR * (1 + taper + taper**2) / max(1 + taper, 1e-6)
    sweep_deg = params['wing_sweep']
    sweep = np.deg2rad(sweep_deg)
    dihedral_rad = np.deg2rad(params['wing_dihedral'])
    tc    = params['wing_thickness']
    wing_z_abs = params['wing_z'] * fr_z

    has_v = params['has_vtail'] > 0.5
    has_h = params['has_htail'] > 0.5
    cant_deg = params.get('vtail_cant', 0.0)
    cant_r = np.deg2rad(cant_deg)

    if has_v:
        vt_h_raw = params['vtail_size'] * L * 1.2
        vt_h_cap = max(min(H - fr_z, 0.22 * L), fr_z)
        vt_h_eff = min(vt_h_raw, vt_h_cap)
        vt_c_eff = min(params['vtail_size'] * L * 1.1, 1.4 * vt_h_eff)
        vt_sweep_t = np.tan(np.deg2rad(params['vtail_sweep']))
        Sv_raw   = vt_h_eff * vt_c_eff
    else:
        vt_h_eff = 0.0; vt_c_eff = 0.0; vt_sweep_t = 0.0; Sv_raw = 0.0

    Sv_vert_eff = Sv_raw * (np.cos(cant_r) ** 2)
    Sv_horz_eff = Sv_raw * (np.sin(cant_r) ** 2)

    if has_h:
        wing_half = params['wing_span'] * W / 2.0
        hs = params['htail_span'] * wing_half
        b_h = 2.0 * hs
        c_h = min(params['htail_chord'] * L, hs * 0.55)
    else:
        hs = 0.0; b_h = 0.0; c_h = 0.0
    Sh       = b_h * c_h
    Sh_total = Sh + Sv_horz_eff
    Sv_total = Sv_vert_eff

    tail_type = classify_tail(params)

    n_eng = int(np.clip(round(params['n_engines_norm']), 1, 4))
    n_user = int(user_spec.get('n_engines', 0) or 0)
    if n_user == 0:
        n_eng = int(np.clip(n_eng, cp['min_eng'], cp['max_eng']))

    eL_m  = params['engine_length'] * L
    eS_m  = params['engine_size']   * min(H, W)
    eng_l_cap = float(user_spec.get('engine_l', 0) or 0)
    eng_h_cap = float(user_spec.get('engine_h', 0) or 0)
    eng_w_cap = float(user_spec.get('engine_w', 0) or 0)
    if eng_l_cap > 0: eL_m = min(eL_m, eng_l_cap)
    if eng_h_cap > 0 and eng_w_cap > 0:
        eS_m = min(eS_m, min(eng_h_cap, eng_w_cap))

    A_cross = np.pi * fr_y * fr_z
    V_fuse_gross = A_cross * max(fl - 0.5*(nose_len+tail_len), 0.0) + \
                   A_cross * 0.5 * (nose_len + tail_len) * 0.6
    V_fuse_usable = VOL_PACK_EFF * V_fuse_gross

    a, c_el = fr_y, fr_z
    h_el = ((a - c_el) / max(a + c_el, 1e-6)) ** 2
    P_el = np.pi * (a + c_el) * (1.0 + 3.0*h_el / (10.0 + np.sqrt(max(4.0 - 3.0*h_el, 0.0))))
    Swet_fuse = P_el * (fl - 0.5*(nose_len+tail_len)) + \
                P_el * 0.5 * (nose_len + tail_len) * 0.9

    Swet_wing = 2.03 * Sref
    Swet_vt   = 2.04 * Sv_raw
    Swet_ht   = 2.04 * Sh
    Swet_eng  = n_eng * 2.0 * np.pi * (eS_m/2.0) * eL_m * 1.3
    Swet_tot  = Swet_fuse + Swet_wing + Swet_vt + Swet_ht + Swet_eng

    Cf = 0.0035 if cls == 'airliner' else 0.0042
    fineness = fl / max(fd, 1e-6)
    FF_fuse  = 1.0 + 60.0/max(fineness**3, 1.0) + fineness/400.0
    FF_wing  = 1.0 + 2.7*tc + 100.0*tc**4
    FF_tail  = 1.3
    FF_eng   = 1.35
    CD0 = Cf * (FF_fuse*Swet_fuse + FF_wing*Swet_wing +
                FF_tail*(Swet_vt + Swet_ht) +
                FF_eng*Swet_eng) / max(Sref, 1e-6)
    CD0 *= 1.00 if tail_type == 'flying_wing' else 1.08

    Wt = m_target * G_ACCEL
    q   = 0.5 * RHO_CRUISE * V*V
    CLc = Wt / max(q * Sref, 1e-6)

    e0  = 0.85 * (1.0 - 2.0*(fd/max(b,1e-6))**2)
    if tail_type == 'flying_wing': e0 = 0.92
    e0  = float(np.clip(e0, 0.60, 0.92))
    CDi = CLc*CLc / max(np.pi * AR * e0, 1e-6)

    M      = V / A_SOUND
    cos_L  = max(np.cos(sweep), 0.30)
    Mcrit  = KORN_KAPPA / cos_L - tc/(cos_L**2) - CLc/(10.0*cos_L**3)
    Mcrit  = float(np.clip(Mcrit, 0.55, 0.92))
    dM     = max(0.0, M - Mcrit)
    CDw    = 20.0 * dM**3 + 80.0 * dM**4

    CD = CD0 + CDi + CDw
    D  = q * Sref * CD
    LD = CLc / max(CD, 1e-6)

    T_cruise = T_total * (RHO_CRUISE / RHO_SL) ** 0.7
    thrust_margin = T_cruise / max(D, 1e-6)

    c_sp   = G_ACCEL * TSFC
    ff_req = 1.0 - np.exp(-R_req * c_sp / max(V * LD, 1e-6))
    ff_req = float(np.clip(ff_req, 0.0, 0.99))
    ff_cap_vol = (FUEL_VOL_FRAC_MAX * V_fuse_usable * RHO_FUEL) / max(m_target, 1e-6)
    ff_used = float(np.clip(min(ff_req, FF_MAX, ff_cap_vol), 0.0, 0.99))
    fuel_mass  = ff_used * m_target
    R_actual   = (V / c_sp) * LD * np.log(1.0 / max(1.0 - ff_used, 1e-6))
    range_ratio = R_actual / max(R_req, 1e-6)
    feasible_fuel = ff_used >= ff_req * 0.99

    structure_mass_geom = Swet_tot * areal_rho
    structure_mass_min  = STRUCT_MIN_FRAC * m_target
    structure_mass      = max(structure_mass_geom, structure_mass_min)
    structure_floored   = structure_mass_geom < structure_mass_min

    engine_mass   = T_total / G_ACCEL / 6.0
    systems_mass  = SYSTEMS_FRAC * m_target
    fixed         = structure_mass + engine_mass + fuel_mass + systems_mass
    implied_payload_raw = m_target - fixed

    V_fuel_used        = fuel_mass / RHO_FUEL
    V_avail_for_pl     = max(V_fuse_usable - V_fuel_used, 0.0)
    max_payload_by_vol = V_avail_for_pl * RHO_PAYLOAD
    implied_payload    = max(0.0, min(implied_payload_raw, max_payload_by_vol))
    payload_frac       = implied_payload / max(m_target, 1e-6)

    V_need_total = V_fuel_used + max(implied_payload, 0.0) / RHO_PAYLOAD
    vol_score = 0.0 if V_fuse_usable <= 0.0 else \
                float(np.clip(V_fuse_usable / max(V_need_total, 1e-6), 0.0, 1.0))

    cl_ok = CLc < CL_MAX * 0.90
    stall_score = float(np.clip(1.0 - CLc / (CL_MAX*0.9), 0, 1))

    halfspan_load = Wt / 2.0
    y_bar  = (2.0 * b) / (3.0 * np.pi)
    M_root = ULT_N * halfspan_load * y_bar
    t_root = tc * cR
    S_mod  = WINGBOX_EFF * cR * t_root**2 / 6.0
    stress = M_root / max(S_mod, 1e-6)
    yield_pa = YIELD_MPA * 1e6
    struct_ok    = stress < yield_pa
    struct_score = float(np.clip(yield_pa / max(stress, 1e-6), 0, 1))

    x_LE_root = (0.5 - params['wing_x']) * fl
    y_MAC = (b / 6.0) * (1.0 + 2.0*taper) / max(1.0 + taper, 1e-6)
    x_wing_ac = x_LE_root - y_MAC * np.tan(sweep) - 0.25 * MAC

    # ---- Engine x-position (matches dataset.voxelize_jet) ----
    if n_eng == 1:
        x_eng = x_tail + eL_m * 0.30   # rear-mounted single
    else:
        x_eng = (0.5 - params['engine_x']) * fl

    x_fuse = 0.0
    x_htac = -fl/2 + 0.15*fl if (has_h or (has_v and cant_deg > 20)) else x_wing_ac
    x_vtac = -fl/2 + 0.20*fl if has_v else 0.0

    masses_full = np.array([structure_mass*0.55, structure_mass*0.30,
                            structure_mass*0.15, engine_mass,
                            fuel_mass, systems_mass])
    xs          = np.array([x_fuse, x_wing_ac, x_htac,
                            x_eng,  x_wing_ac*0.9, x_fuse])
    x_cg_full  = _cg_x(masses_full, xs)
    masses_empty = masses_full.copy(); masses_empty[4] = 0.0
    x_cg_empty = _cg_x(masses_empty, xs)

    if Sh_total > 1e-6:
        Vh = (Sh_total / max(Sref, 1e-6)) * (x_wing_ac - x_htac) / max(MAC, 1e-6)
    else:
        Vh = 0.0
    if tail_type == 'flying_wing':
        Vh = max(Vh, 0.2 * (sweep_deg / 45.0))

    x_np = x_wing_ac - 0.35 * MAC * Vh
    sm_full  = (x_cg_full  - x_np) / max(MAC, 1e-6)
    sm_empty = (x_cg_empty - x_np) / max(MAC, 1e-6)
    stab_score = 0.5*_static_margin_score(sm_full) + 0.5*_static_margin_score(sm_empty)

    cg_offset_mac  = abs(x_cg_full - x_wing_ac) / max(MAC, 1e-6)
    cg_place_score = float(np.clip(1.0 - cg_offset_mac / CG_PLACE_TOL_MAC, 0.0, 1.0))

    if Sv_total > 1e-6:
        Lv = x_wing_ac - x_vtac
        Vv = (Sv_total * max(Lv, 1e-3)) / max(Sref * b, 1e-6)
    else:
        Vv = 0.0
    if tail_type == 'flying_wing':
        Vv = max(Vv, 0.015 * (sweep_deg / 45.0))
    vtail_vol_score = _vtail_volume_score(Vv)

    tip_TE_x = x_LE_root - (b/2)*np.tan(sweep) - cT
    box_ok = (fl <= fuse_L + 1e-3 and
              (2*fr_y) <= fuse_W + 1e-3 and (2*fr_z) <= fuse_H + 1e-3 and
              b <= W + 1e-3 and tip_TE_x > -L/2 - 0.05*L)
    box_score = 1.0 if box_ok else 0.4

    fuse_len_ratio = fl / max(fuse_L, 1e-6)
    fuse_len_score = 1.0 if fuse_len_ratio >= MIN_FUSE_LEN_FRAC \
                     else max(0.0, fuse_len_ratio / MIN_FUSE_LEN_FRAC)

    if thrust_margin < 1.0:           tm_score = max(0.0, thrust_margin * 0.5)
    elif thrust_margin < T_MARGIN_LO: tm_score = 0.5 + 0.5*(thrust_margin-1.0)/(T_MARGIN_LO-1.0)
    elif thrust_margin <= T_MARGIN_HI:tm_score = 1.0
    else:                             tm_score = max(0.25, 1.0 - 0.4*(thrust_margin-T_MARGIN_HI))

    target_pf = 0.20 if cls == 'airliner' else 0.12
    payload_target = user_spec.get('payload_mass', None)
    if payload_target is not None and payload_target > 0:
        if implied_payload < payload_target:
            pay_score = (implied_payload / payload_target) ** 1.5
        else:
            over = (implied_payload - payload_target) / payload_target
            pay_score = float(np.clip(1.0 - 0.3*over, 0.4, 1.0))
    else:
        pay_score = float(np.clip(payload_frac / target_pf, 0.0, 1.0))
    pay_score = float(np.clip(pay_score, 0.0, 1.0))

    fuel_score = float(np.clip(1.0 - ff_req / FF_MAX, 0.0, 1.0))

    mount_score, worst_pen, is_fuse_mount, eng_centers = _engine_attachment_score(
        params, fl, fr_y, fr_z, fr_eq, nose_len, tail_len,
        b, cR, taper, sweep, dihedral_rad,
        wing_z_abs, tc, W, H, eS_m, eL_m, n_eng)
    mount_pen_frac = worst_pen / max(eS_m, 1e-6)
    mount_firm     = mount_pen_frac >= MOUNT_MIN_PEN_FRAC
    mount_touching = (mount_pen_frac >= 0.0) and (not mount_firm)
    mount_floating = mount_pen_frac < 0.0

    ht_mount_score   = 1.0
    ht_mount_pen     = 1e9
    ht_mount_pen_f   = 1e9
    ht_mount_firm    = True
    ht_mount_touch   = False
    ht_floating      = False
    ht_thick_r_m     = 0.0
    ht_z_abs_m       = 0.0
    ht_attached_to   = 'none'

    if has_h:
        fin_is_vertical = has_v and cant_deg < 20.0 and vt_h_eff > 0.0
        if fin_is_vertical:
            ref_h = vt_h_eff
            htz_used = params['htail_z']
        else:
            ref_h = fr_z
            htz_used = min(params['htail_z'], 0.15)

        ht_z_abs_m   = fr_z * 0.3 + htz_used * ref_h
        ht_chord_r_m = min(params['htail_chord'] * L, hs * 0.55)
        ht_x_LE_r_m  = x_tail + ht_chord_r_m
        ht_thick_r_m = max(min(L, H, W) / 70.0, 0.05)

        ht_mount_pen = _htail_attachment_pen(
            fl, fr_eq, fr_z, nose_len, tail_len,
            ht_x_LE_r_m, ht_chord_r_m, ht_thick_r_m, ht_z_abs_m,
            has_v, cant_deg, vt_h_eff, vt_c_eff, vt_sweep_t,
            x_tail)
        ht_mount_score = _htail_mount_score(ht_mount_pen, ht_thick_r_m)
        ht_mount_pen_f = ht_mount_pen / max(ht_thick_r_m, 1e-6)
        ht_mount_firm  = ht_mount_pen_f >= HT_MOUNT_MIN_PEN_FRAC
        ht_mount_touch = (ht_mount_pen_f >= 0.0) and (not ht_mount_firm)
        ht_floating    = ht_mount_pen_f < 0.0

        x_mid = ht_x_LE_r_m - ht_chord_r_m / 2.0
        r_at_mid = _fuselage_radius_at(np.clip(x_mid, -fl/2, fl/2),
                                        fl, fr_eq, nose_len, tail_len)
        fuse_touch = (r_at_mid + ht_thick_r_m) >= abs(ht_z_abs_m)
        if fuse_touch:
            ht_attached_to = 'fuselage'
        elif fin_is_vertical and (fr_z*0.3 <= ht_z_abs_m <= fr_z*0.3 + vt_h_eff):
            ht_attached_to = 'fin'
        else:
            ht_attached_to = 'floating'

    ground_score = _ground_clearance_score(eng_centers, is_fuse_mount, eS_m, fr_z)

    ld_score = float(np.clip(LD / LD_TARGET, 0.0, 1.0))
    ar_score = float(np.clip(AR / AR_TARGET, 0.0, 1.0))

    chord_ratio = cR / max(fuse_L, 1e-6)
    bL          = b / max(L, 1e-6)
    chord_real_score = _chord_realism(chord_ratio)
    cl_real_score    = _cl_realism(CLc)
    span_real_score  = _span_realism(bL)

    WoS = m_target / max(Sref, 1e-6)
    wing_load_score = _wing_loading_realism(WoS, cls)

    V_wing_est = 0.685 * tc * (b/3.0) * cR*cR * (1.0 + taper + taper*taper)
    V_vt_est   = 0.10 * vt_h_eff * vt_c_eff * vt_c_eff if has_v else 0.0
    V_ht_est   = 0.10 * b_h * c_h * c_h if has_h else 0.0
    V_eng_est  = n_eng * eL_m * eS_m * eS_m
    V_total_est = V_fuse_gross + V_wing_est + V_vt_est + V_ht_est + V_eng_est
    V_ref_m3    = m_target / DENSITY_REF_KG_M3
    shape_ratio = V_total_est / max(V_ref_m3, 1e-3)
    block_score = _block_economy(shape_ratio)

    tail_config_score = 1.0
    if cls == 'airliner' and cp['prefer_tube_and_wing']:
        if tail_type == 'flying_wing': tail_config_score = 0.92
        elif tail_type in ('conventional','t_tail','cruciform','v_tail'): tail_config_score = 1.0
        else: tail_config_score = 0.85
    if tail_type == 'flying_wing' and sweep_deg < 25.0:
        tail_config_score *= 0.5

    fitness = (
        0.16 * min(1.0, range_ratio) +
        0.05 * tm_score +
        0.07 * ld_score +
        0.04 * ar_score +
        0.03 * stall_score +
        0.06 * stab_score +
        0.05 * struct_score +
        0.09 * pay_score +
        0.03 * fuel_score +
        0.02 * box_score +
        0.06 * mount_score +
        0.03 * ht_mount_score +
        0.03 * vtail_vol_score +
        0.05 * vol_score +
        0.03 * ground_score +
        0.03 * fuse_len_score +
        0.03 * cg_place_score +
        0.04 * chord_real_score +
        0.03 * cl_real_score +
        0.03 * span_real_score +
        0.05 * block_score +
        0.05 * wing_load_score
    ) * tail_config_score

    if not cl_ok:           fitness *= 0.40
    if not struct_ok:       fitness *= 0.35
    if thrust_margin < 1.0: fitness *= (0.4 + 0.6*thrust_margin)
    if not feasible_fuel:
        sh = 1.0 - ff_used / max(ff_req, 1e-6)
        fitness *= max(0.35, 1.0 - 0.55 * sh)
    if implied_payload_raw < 0:
        fn = implied_payload_raw / max(m_target, 1e-6)
        fitness *= max(0.25, 1.0 + 2.0 * fn)
    elif payload_frac < 0.03:
        fitness *= 0.7
    if not box_ok: fitness *= 0.20
    if structure_floored:
        sh = 1.0 - structure_mass_geom / max(structure_mass_min, 1e-6)
        fitness *= max(0.55, 1.0 - 0.4 * sh)
    if fuse_len_ratio < MIN_FUSE_LEN_FRAC:
        fitness *= (0.55 + 0.45 * fuse_len_score)

    if Sv_total > 1e-6 and Vv < VV_MIN and tail_type != 'flying_wing':
        fitness *= 0.65
    if Sv_total > 1e-6 and Vv > VV_MAX:
        over = (Vv - VV_MAX) / VV_MAX
        fitness *= max(0.45, 1.0 - 0.6 * over)
    sv_ratio = Sv_raw / max(Sref, 1e-6)
    if sv_ratio > SV_OVER_SREF_MAX:
        over = sv_ratio / SV_OVER_SREF_MAX - 1.0
        fitness *= max(0.35, 1.0 - 0.9 * over)
    sh_ratio = Sh / max(Sref, 1e-6)
    if sh_ratio > 0.22:
        over = sh_ratio / 0.22 - 1.0
        fitness *= max(0.45, 1.0 - 0.7 * over)

    if mount_floating:
        g = min(-mount_pen_frac, MOUNT_MAX_GAP_FRAC) / MOUNT_MAX_GAP_FRAC
        fitness *= max(0.02, 0.10 - 0.08 * g)
    elif mount_touching:
        t = mount_pen_frac / MOUNT_MIN_PEN_FRAC
        fitness *= (0.20 + 0.25 * t)
    elif mount_score < 0.95:
        fitness *= (0.40 + 0.60 * mount_score) ** 2

    if has_h:
        if ht_floating:
            g = min(-ht_mount_pen_f, HT_MOUNT_MAX_GAP_FRAC) / HT_MOUNT_MAX_GAP_FRAC
            fitness *= max(0.02, 0.10 - 0.08 * g)
        elif ht_mount_touch:
            t = ht_mount_pen_f / HT_MOUNT_MIN_PEN_FRAC
            fitness *= (0.20 + 0.25 * t)
        elif ht_mount_score < 0.95:
            fitness *= (0.40 + 0.60 * ht_mount_score) ** 2

    if ground_score < 0.5: fitness *= (0.4 + 0.6 * ground_score)
    if not (STATIC_MARGIN_LO <= sm_full  <= STATIC_MARGIN_HI): fitness *= 0.80
    if not (STATIC_MARGIN_LO <= sm_empty <= STATIC_MARGIN_HI): fitness *= 0.80
    if cg_offset_mac > CG_PLACE_TOL_MAC:
        over = cg_offset_mac - CG_PLACE_TOL_MAC
        fitness *= max(0.55, 1.0 - 0.6 * over)

    if chord_ratio > CHORD_RATIO_HARD:
        over = chord_ratio - CHORD_RATIO_HARD
        fitness *= max(0.15, 1.0 - 1.6 * over)
    if CLc < CL_CRUISE_HARD_LO:
        fitness *= max(0.10, (CLc / CL_CRUISE_HARD_LO) ** 2)
    if shape_ratio > SHAPE_RATIO_HARD:
        over = shape_ratio - SHAPE_RATIO_HARD
        fitness *= max(0.08, 1.0 - 0.25 * over)
    if bL > 2.0 and tail_type != 'flying_wing':
        over = bL - 2.0
        fitness *= max(0.20, 1.0 - 0.4 * over)

    wl_hard = WING_LOAD_HARD_LO.get(cls, 130.0)
    if WoS < wl_hard:
        fitness *= max(0.05, (WoS / wl_hard) ** 2)

    if tail_type == 'h_only':
        fitness *= 0.10
    elif tail_type == 'v_only':
        fitness *= 0.55

    return float(fitness), {
        'vehicle_class': cls,
        'tail_type':     tail_type,
        'tsfc_used': float(TSFC), 'ff_max_used': float(FF_MAX),
        'ld_target': float(LD_TARGET), 'ar_target': float(AR_TARGET),
        'range_km': float(R_actual / 1000.0),
        'range_required_km': float(R_req / 1000.0),
        'range_ratio': float(range_ratio),
        'fuel_fraction_req': float(ff_req),
        'fuel_fraction_used': float(ff_used),
        'fuel_vol_cap_ff': float(ff_cap_vol),
        'LD': float(LD), 'CD0': float(CD0), 'CDi': float(CDi), 'CDw': float(CDw),
        'CL_cruise': float(CLc), 'mach': float(M), 'mach_crit': float(Mcrit),
        'thrust_margin': float(thrust_margin), 'drag_N': float(D),
        'wing_area_m2': float(Sref), 'AR': float(AR), 'MAC_m': float(MAC),
        'wing_loading_kg_m2': float(WoS),
        'wing_load_score':    float(wing_load_score),
        'wing_LE_root_x_m': float(x_LE_root), 'wing_AC_x_m': float(x_wing_ac),
        'wing_AC_frac_fl': float(0.5 - x_wing_ac / max(fl, 1e-6)),
        'y_MAC_m': float(y_MAC),
        'cg_full_x_m': float(x_cg_full), 'cg_empty_x_m': float(x_cg_empty),
        'neutral_point_x_m': float(x_np),
        'cg_offset_mac': float(cg_offset_mac),
        'cg_place_score': float(cg_place_score),
        'structure_mass_kg': float(structure_mass),
        'structure_mass_geom_kg': float(structure_mass_geom),
        'structure_floored': bool(structure_floored),
        'engine_mass_kg': float(engine_mass),
        'fuel_mass_kg': float(fuel_mass),
        'systems_mass_kg': float(systems_mass),
        'implied_payload_raw_kg': float(implied_payload_raw),
        'implied_payload_kg': float(implied_payload),
        'max_payload_by_vol_kg': float(max_payload_by_vol),
        'payload_fraction': float(payload_frac),
        'static_margin_full': float(sm_full),
        'static_margin_empty': float(sm_empty),
        'vtail_volume_coef': float(Vv),
        'vtail_cant_deg': float(cant_deg),
        'sh_effective_m2': float(Sh_total),
        'sv_effective_m2': float(Sv_total),
        'fuse_ry_m': float(fr_y),
        'fuse_rz_m': float(fr_z),
        'engine_length_m': float(eL_m),
        'engine_size_m': float(eS_m),
        'engine_x_m': float(x_eng),
        'fuselage_volume_m3': float(V_fuse_gross),
        'fuselage_usable_m3': float(V_fuse_usable),
        'fuel_volume_m3': float(V_fuel_used),
        'volume_required_m3': float(V_need_total),
        'volume_score': float(vol_score),
        'fuse_len_ratio': float(fuse_len_ratio),
        'wing_root_stress_MPa': float(stress / 1e6),
        'n_engines': int(n_eng),
        'n_engines_forced_by_user': bool(n_user > 0),
        'has_vtail': bool(has_v), 'has_htail': bool(has_h),
        'box_ok': bool(box_ok), 'feasible_fuel': bool(feasible_fuel),
        'engine_mount_score': float(mount_score),
        'engine_mount_pen_m': float(worst_pen),
        'engine_mount_pen_frac': float(mount_pen_frac),
        'engine_mount_type': 'fuselage' if is_fuse_mount else 'podded',
        'engine_mount_firm': bool(mount_firm),
        'engine_mount_touching_only': bool(mount_touching),
        'engine_floating': bool(mount_floating),
        'htail_mount_score': float(ht_mount_score),
        'htail_mount_pen_m': float(ht_mount_pen if has_h else 0.0),
        'htail_mount_pen_frac': float(ht_mount_pen_f if has_h else 0.0),
        'htail_mount_firm': bool(ht_mount_firm),
        'htail_mount_touching_only': bool(ht_mount_touch),
        'htail_floating': bool(ht_floating),
        'htail_attached_to': ht_attached_to,
        'ground_clearance_score': float(ground_score),
        'chord_ratio_fuse': float(chord_ratio),
        'span_to_length':   float(bL),
        'shape_density_ratio': float(shape_ratio),
        'voxel_volume_estimate_m3': float(V_total_est),
        'block_score': float(block_score),
        'chord_realism_score': float(chord_real_score),
        'cl_realism_score':    float(cl_real_score),
        'span_realism_score':  float(span_real_score),
        'fitness': float(fitness),
    }