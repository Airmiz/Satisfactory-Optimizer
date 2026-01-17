from fractions import Fraction
from math import gcd
from functools import reduce
import itertools

def lcm(a, b):
    return a // gcd(a, b) * b

def lcm_many(nums):
    nums = [abs(n) for n in nums if n != 0]
    return reduce(lcm, nums, 1)

def gcd_many(nums):
    nums = [abs(n) for n in nums if n != 0]
    return reduce(gcd, nums, 0)

def to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(1_000_000)
    raise TypeError(f"Unsupported number type: {type(x)}")

def gauss_jordan_solve(A, b):
    m, n = len(A), len(A[0])
    M = [A[i][:] + [b[i]] for i in range(m)]

    row = 0
    pivots = [-1] * n
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue

        M[row], M[pivot] = M[pivot], M[row]
        pv = M[row][col]
        M[row] = [v / pv for v in M[row]]

        for r in range(m):
            if r == row:
                continue
            factor = M[r][col]
            if factor != 0:
                M[r] = [M[r][c] - factor * M[row][c] for c in range(n + 1)]

        pivots[col] = row
        row += 1
        if row == m:
            break

    for r in range(m):
        if all(M[r][c] == 0 for c in range(n)) and M[r][n] != 0:
            raise ValueError("No solution (inconsistent constraints).")

    x = [Fraction(0) for _ in range(n)]
    for col in range(n):
        r = pivots[col]
        if r != -1:
            x[col] = M[r][n]
    return x

def lp_solve_nonnegative_min_sum(Aeq, beq):
    """
    minimize sum(x)
    s.t. Aeq x = beq
         x >= 0
    """
    # Prefer SciPy if available
    try:
        import numpy as np
        from scipy.optimize import linprog

        A = np.array([[float(v) for v in row] for row in Aeq], dtype=float)
        b = np.array([float(v) for v in beq], dtype=float)
        c = np.ones(A.shape[1], dtype=float)
        bounds = [(0, None)] * A.shape[1]

        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
        if not res.success:
            raise ValueError(f"LP failed: {res.message}")

        return [Fraction(v).limit_denominator(1_000_000) for v in res.x.tolist()]

    except ModuleNotFoundError:
        pass

    # Fallback (small problems): enumerate basic feasible solutions
    m = len(Aeq)
    n = len(Aeq[0])
    best_x = None
    best_obj = None

    for basis in itertools.combinations(range(n), m):
        Ab = [[Aeq[i][j] for j in basis] for i in range(m)]
        try:
            xb = gauss_jordan_solve(Ab, beq)
        except ValueError:
            continue

        x = [Fraction(0) for _ in range(n)]
        feasible = True
        for k, j in enumerate(basis):
            x[j] = xb[k]
            if x[j] < 0:
                feasible = False
                break
        if not feasible:
            continue

        for i in range(m):
            lhs = sum(Aeq[i][j] * x[j] for j in range(n))
            if lhs != beq[i]:
                feasible = False
                break
        if not feasible:
            continue

        obj = sum(x)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_x = x

    if best_x is None:
        raise ValueError("No feasible nonnegative solution found (x >= 0).")

    return best_x

def solve_satisfactory_clean_plan(
    recipes,
    target_item,
    target_rate_per_min,
    raw_items=None,
    allow_extra_outputs=None,
    objective_weights=None,
):
    """
    recipes: dict name -> {
        "building": "Constructor"/"Assembler"/... (optional),
        "time": seconds_per_craft,
        "in":  {item: qty_per_craft},
        "out": {item: qty_per_craft},
    }

    target_rate_per_min: desired output in items/min (int/float/Fraction)
    raw_items: items allowed net-negative (mined/pumped)
    allow_extra_outputs: items allowed net-positive (byproducts). target_item always allowed.
    objective_weights: dict recipe_name -> weight for objective.
        Default objective minimizes total buildings (sum x). Set weights to prefer/avoid certain machines/recipes.

    Returns:
      buildings_100: recipe_name -> Fraction buildings at 100% clock
      scaled_integer: recipe_name -> int ratio (nice “clean” whole-building proportions)
      net_per_min: item -> Fraction net items/min
    """
    raw_items = set(raw_items or [])
    target_rate_per_min = to_fraction(target_rate_per_min)

    if allow_extra_outputs is None:
        allow_extra_outputs = {target_item}
    else:
        allow_extra_outputs = set(allow_extra_outputs) | {target_item}

    recipe_names = list(recipes.keys())
    n = len(recipe_names)

    # Collect items
    items = set([target_item])
    for r in recipes.values():
        items |= set(r.get("in", {}).keys())
        items |= set(r.get("out", {}).keys())

    # Build flows per ONE BUILDING at 100% clock, in items/min.
    # crafts_per_min = 60 / time_seconds
    flow = {name: {} for name in recipe_names}
    for name, r in recipes.items():
        t = to_fraction(r.get("time", 1))
        crafts_per_min = Fraction(60, 1) / t

        for it, q in r.get("out", {}).items():
            flow[name][it] = flow[name].get(it, Fraction(0)) + to_fraction(q) * crafts_per_min
        for it, q in r.get("in", {}).items():
            flow[name][it] = flow[name].get(it, Fraction(0)) - to_fraction(q) * crafts_per_min

    # Balanced items: not raw, and not allowed net-positive (target/byproducts)
    balanced_items = [it for it in sorted(items)
                      if it not in raw_items and it not in allow_extra_outputs]

    eq_items = balanced_items + [target_item]

    Aeq = []
    beq = []
    for it in eq_items:
        Aeq.append([flow[name].get(it, Fraction(0)) for name in recipe_names])
        beq.append(Fraction(0) if it != target_item else target_rate_per_min)

    # Objective: minimize sum(w_i * x_i). We implement this by scaling columns.
    # linprog expects c^T x, but our lp solver currently minimizes sum(x).
    # Trick: change variable y_i = w_i * x_i => x_i = y_i / w_i
    # Then minimize sum(y_i) with adjusted A columns: A[:,i] / w_i.
    weights = {name: to_fraction(1) for name in recipe_names}
    if objective_weights:
        for k, w in objective_weights.items():
            weights[k] = to_fraction(w)

    Aeq_weighted = []
    for row in Aeq:
        Aeq_weighted.append([row[i] / weights[recipe_names[i]] for i in range(n)])

    y = lp_solve_nonnegative_min_sum(Aeq_weighted, beq)
    x = [y[i] / weights[recipe_names[i]] for i in range(n)]  # buildings at 100%

    buildings_100 = {recipe_names[i]: x[i] for i in range(n)}

    # Net per minute
    net = {it: Fraction(0) for it in items}
    for name, bcount in buildings_100.items():
        for it, coeff in flow[name].items():
            net[it] += coeff * bcount

    # Pretty integer ratio
    denoms = [v.denominator for v in x]
    scale = lcm_many(denoms)
    ints = [int(v * scale) for v in x]
    g = gcd_many(ints)
    if g > 0:
        ints = [k // g for k in ints]
    scaled_integer = {recipe_names[i]: ints[i] for i in range(n)}

    return buildings_100, scaled_integer, net


if __name__ == "__main__":
    # Imagine: Iron Ingot -> Iron Plate -> Reinforced Plate (toy chain)
    recipes = {
        "Iron Ingot (Smelter)": {
            "building": "Smelter",
            "time": 2,
            "in": {"Iron Ore": 1},
            "out": {"Iron Ingot": 1},
        },
        "Iron Plate (Constructor)": {
            "building": "Constructor",
            "time": 6,
            "in": {"Iron Ingot": 3},
            "out": {"Iron Plate": 2},
        },
        "Iron Rod (Constructor)": {
            "building": "Constructor",
            "time": 4,
            "in": {"Iron Ingot": 1},
            "out": {"Iron Rod": 1},        
        },
        "Screw (Constructor)": {
            "building": "Constructor",
            "time": 6,
            "in": {"Iron Rod": 1},
            "out": {"Screw": 4},
        },
        "Reinforced Plate (Assembler)": {
            "building": "Assembler",
            "time": 12,
            "in": {"Iron Plate": 6, "Screw": 12},
            "out": {"Reinforced Iron Plate": 1},
        },
        "Rotor (Assembler)": {
            "building": "Assembler",
            "time": 15,
            "in": {"Iron Rod": 5, "Screw": 25 },
            "out": {"Rotor": 1}
        },
        "Smart Plating (Assembler)": {
            "building": "Assembler",
            "time": 30,
            "in": {"Reinforced Iron Plate": 2, "Rotor": 2},
            "out": {"Smart Plating": 2}
        },
    }

    buildings, ratio, net = solve_satisfactory_clean_plan(
        recipes,
        target_item="Reinforced Iron Plate",
        target_rate_per_min=10,         # want 10 / min
        raw_items={"Iron Ore"},         # allow ore to be imported
        # objective_weights={"Reinforced Plate (Assembler)": 10}  # example: "avoid" a recipe by weighting it higher
    )

    print("Buildings at 100% clock (can be fractional):")
    for k, v in buildings.items():
        if v != 0:
            print(f"  {k}: {v}")

    print("\nClean integer proportions:")
    for k, v in ratio.items():
        if v != 0:
            print(f"  {k}: {v}")

    print("\nNet flows (items/min):")
    for it, v in sorted(net.items()):
        if v != 0:
            print(f"  {it}: {v}")
