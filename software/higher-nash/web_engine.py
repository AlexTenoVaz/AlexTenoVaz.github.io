# Motor web para la demostración del Teorema 3.3.
# Usa las funciones de Funciones.py sin exponer un editor de código al usuario.

import json
import random
import sympy as sp
from sympy import symbols, I, sqrt, limit, oo, zoo, latex

from Funciones import (
    default_row_vec, default_col_vec, curve_on_ED, curva, orden_entero,
    jacobian_matrix, solve_matching, next_matching, gauss_custom, order_pol
)

x, y, z, t = symbols("x y z t")

def _finite(expr):
    if expr in (oo, -oo, zoo, I*oo, -I*oo):
        return False
    return expr.is_finite is not False

def run_demo(jac_order=5):
    """Ejecuta el caso de demostración publicado en la tesis."""
    # Caso del ejemplo: f=x^2+y^3+z^4, divisor x+i*z, carta zy.
    f = x**2 + y**3 + z**4
    divisor = x + I*z
    number_of_vars = 3
    indices = [2, 1]

    random.seed(0)

    row_vec = default_row_vec(number_of_vars, jac_order)
    col_vec = default_col_vec(number_of_vars, jac_order)

    curvas_sobre_div = curve_on_ED(
        divisor, indices[-1], t**2, t + 1, t - 1
    )
    curvas_bajadas = [curva(c, indices) for c in curvas_sobre_div]

    # En el script original este resultado se fija explícitamente.
    exp_curves = [[orden_entero(par) for par in cur] for cur in curvas_bajadas]
    exp_curves = [[2, 2, 2]]

    M = jacobian_matrix(
        f, row_vec, col_vec, None, number_of_vars, jac_order
    )
    M = gauss_custom(M)
    M = M.subs({x: t**2, y: t**2, z: t**2})
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
        vals = []
        for curve in curvas_bajadas:
            vals.append(sp.simplify(det.subs({
                x: curve[0], y: curve[1], z: curve[2]
            })))
        evaluations.append(vals)

    comparison = None
    found = False
    for a in range(len(evaluations)):
        for b in range(a + 1, len(evaluations)):
            M11, M12 = evaluations[a]
            M21, M22 = evaluations[b]

            try:
                r21_11 = limit(M21 / M11, t, 0)
                r22_12 = limit(M22 / M12, t, 0)
                r11_21 = limit(M11 / M21, t, 0)
                r12_22 = limit(M12 / M22, t, 0)
            except Exception:
                continue

            if not all(_finite(q) for q in (r21_11, r22_12, r11_21, r12_22)):
                continue

            if r21_11 == r22_12:
                continue

            comparison = {
                "m1": a + 1,
                "m2": b + 1,
                "r21_11": latex(r21_11),
                "r22_12": latex(r22_12),
                "r11_21": latex(r11_21),
                "r12_22": latex(r12_22),
                "different_forward": bool(r21_11 != r22_12),
                "different_reverse": bool(r11_21 != r12_22),
            }
            found = True
            break
        if found:
            break

    return {
        "polynomial": latex(f),
        "divisor": latex(divisor),
        "jac_order": jac_order,
        "chart": "zy",
        "curves": [[latex(e) for e in c] for c in curvas_sobre_div],
        "curves_down": [[latex(e) for e in c] for c in curvas_bajadas],
        "exponents": exp_curves,
        "matrix_shape": [int(M.rows), int(M.cols)],
        "matching_count": len(matchings),
        "matchings": matchings,
        "cost": int(cost1),
        "determinants": [latex(d) for d in determinants],
        "evaluations": [[latex(v) for v in vals] for vals in evaluations],
        "comparison": comparison,
        "conclusion": (
            "El divisor sí aparece en la explosión de Nash "
            f"de orden {jac_order}."
            if comparison and comparison["different_forward"]
            else "No se obtuvo una pareja válida con el criterio del ejemplo."
        ),
    }

def run_demo_json(jac_order=5):
    return json.dumps(run_demo(int(jac_order)), ensure_ascii=False)
