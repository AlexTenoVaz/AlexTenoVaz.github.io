# Adaptador web: mantiene la lógica matemática en Funciones.py
import json
import random
import sympy as sp

from Funciones import (
    x, y, z, t, I, sqrt, oo, zoo, latex, limit,
    default_row_vec, default_col_vec, curve_on_ED, curva, orden_entero,
    jacobian_matrix, solve_matching, next_matching, gauss_custom, order_pol
)

def es_finito(expr):
    if expr in (oo, -oo, zoo, I*oo, -I*oo):
        return False
    return expr.is_finite is not False

def run_demo(jac_order=5):
    """Caso reproducible del ejemplo de la tesis."""
    f = x**2 + y**3 + z**4
    divisor = x + I*z
    number_of_vars = 3
    indices = [2, 1]

    # El script original utiliza estas dos curvas.
    curvas_sobre_div = curve_on_ED(
        divisor, indices[-1], t**2, t + 1, t - 1
    )
    curvas_bajadas = [curva(c, indices) for c in curvas_sobre_div]

    # Se conserva exactamente el valor usado en el script de ejemplo.
    exp_curves_calculated = [
        [orden_entero(par) for par in cur] for cur in curvas_bajadas
    ]
    exp_curves = [[2, 2, 2]]

    row_vec = default_row_vec(number_of_vars, jac_order)
    col_vec = default_col_vec(number_of_vars, jac_order)

    M = jacobian_matrix(
        f, row_vec, col_vec, None, number_of_vars, jac_order
    )
    M = gauss_custom(M)
    M = M.subs({x: t**exp_curves[0][0],
                y: t**exp_curves[0][1],
                z: t**exp_curves[0][2]})
    M = M.applyfunc(order_pol)

    matchings = []
    matching1, cost1 = solve_matching(M)
    cols_selected1 = sorted(int(c[1:]) for _, c in matching1)
    matchings.append(cols_selected1)

    while True:
        _, next_cols, next_matc, _ = next_matching(
            M, prev_cols_list=matchings, cost_target=cost1
        )
        if next_matc is None:
            break
        matchings.append(next_cols)

    jac_mat = jacobian_matrix(
        f, row_vec, col_vec, None, number_of_vars, jac_order
    )
    jac_mat = gauss_custom(jac_mat)

    determinants = [jac_mat[:, cols].det() for cols in matchings]

    evaluations = []
    for det in determinants:
        evaluations.append([
            sp.simplify(det.subs({x: c[0], y: c[1], z: c[2]}))
            for c in curvas_bajadas
        ])

    comparison = None
    for a in range(len(evaluations)):
        for b in range(a + 1, len(evaluations)):
            M11, M12 = evaluations[a]
            M21, M22 = evaluations[b]
            try:
                r21_11 = limit(M21/M11, t, 0)
                r22_12 = limit(M22/M12, t, 0)
                r11_21 = limit(M11/M21, t, 0)
                r12_22 = limit(M12/M22, t, 0)
            except Exception:
                continue

            if not all(es_finito(q) for q in
                       (r21_11, r22_12, r11_21, r12_22)):
                continue
            if r21_11 == r22_12:
                continue

            comparison = {
                "m1": a + 1, "m2": b + 1,
                "r21_11": latex(r21_11),
                "r22_12": latex(r22_12),
                "r11_21": latex(r11_21),
                "r12_22": latex(r12_22),
                "different_forward": bool(r21_11 != r22_12),
                "different_reverse": bool(r11_21 != r12_22),
            }
            break
        if comparison:
            break

    return {
        "polynomial": latex(f),
        "divisor": latex(divisor),
        "jac_order": int(jac_order),
        "chart": "zy",
        "curves": [[latex(e) for e in c] for c in curvas_sobre_div],
        "curves_down": [[latex(e) for e in c] for c in curvas_bajadas],
        "exponents_calculated": exp_curves_calculated,
        "exponents_used": exp_curves,
        "matrix_shape": [int(M.rows), int(M.cols)],
        "matching_count": len(matchings),
        "matchings": matchings,
        "cost": int(cost1),
        "determinants": [latex(d) for d in determinants],
        "evaluations": [[latex(v) for v in vals] for vals in evaluations],
        "comparison": comparison,
        "conclusion": (
            f"The divisor {latex(divisor)} appears in the "
            f"order-{jac_order} Nash modification."
            if comparison and comparison["different_forward"]
            else "No valid pair of matchings was obtained for this order."
        ),
    }

def run_demo_json(jac_order=5):
    return json.dumps(run_demo(int(jac_order)), ensure_ascii=False)
