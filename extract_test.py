import json
import tkinter as tk
from tkinter import ttk, messagebox
from fractions import Fraction
from math import gcd
from functools import reduce
import itertools

# ----------------------------
# Math helpers
# ----------------------------
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
    # JSON numbers may already be int/float; if not:
    return Fraction(str(x)).limit_denominator(1_000_000)

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
    # Prefer SciPy if available (fast)
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

    # Fallback (small/medium problems): enumerate basic feasible solutions
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

        # verify exact
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
    raw_items,
    allow_extra_outputs=None,
):
    raw_items = set(raw_items)
    target_rate_per_min = to_fraction(target_rate_per_min)

    if allow_extra_outputs is None:
        allow_extra_outputs = {target_item}
    else:
        allow_extra_outputs = set(allow_extra_outputs) | {target_item}

    recipe_names = list(recipes.keys())
    n = len(recipe_names)

    # Collect items
    items = {target_item}
    for r in recipes.values():
        items |= set((r.get("in") or {}).keys())
        items |= set((r.get("out") or {}).keys())

    # Flow per ONE BUILDING at 100% clock, in items/min.
    flow = {name: {} for name in recipe_names}
    for name, r in recipes.items():
        t = to_fraction(r.get("time", 1))
        if t == 0:
            # If time=0 exists, treat as invalid for rate math (would be infinite).
            # Skip it safely.
            continue
        crafts_per_min = Fraction(60, 1) / t

        for it, q in (r.get("out") or {}).items():
            flow[name][it] = flow[name].get(it, Fraction(0)) + to_fraction(q) * crafts_per_min
        for it, q in (r.get("in") or {}).items():
            flow[name][it] = flow[name].get(it, Fraction(0)) - to_fraction(q) * crafts_per_min

    balanced_items = [it for it in sorted(items)
                      if it not in raw_items and it not in allow_extra_outputs]

    eq_items = balanced_items + [target_item]
    Aeq = []
    beq = []
    for it in eq_items:
        Aeq.append([flow[name].get(it, Fraction(0)) for name in recipe_names])
        beq.append(Fraction(0) if it != target_item else target_rate_per_min)

    x = lp_solve_nonnegative_min_sum(Aeq, beq)

    buildings_100 = {recipe_names[i]: x[i] for i in range(n)}

    # Net flows per minute
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

# ----------------------------
# Recipe graph narrowing
# ----------------------------
def build_relevant_recipe_subset(all_recipes, target_item, raw_items):
    """
    Backward closure:
      start from target_item, include any recipe that PRODUCES it,
      then add that recipe's inputs as needed items, repeat until hitting raw items.
    This dramatically reduces LP size and avoids unrelated recipes.
    """
    raw_items = set(raw_items)
    produces = {}
    for rname, r in all_recipes.items():
        for out_item in (r.get("out") or {}):
            produces.setdefault(out_item, []).append(rname)

    needed_items = {target_item}
    used_recipes = set()
    queue = [target_item]

    while queue:
        item = queue.pop()
        if item in raw_items:
            continue
        for rname in produces.get(item, []):
            if rname in used_recipes:
                continue
            used_recipes.add(rname)
            for in_item in (all_recipes[rname].get("in") or {}):
                if in_item not in needed_items:
                    needed_items.add(in_item)
                    queue.append(in_item)

    return {rname: all_recipes[rname] for rname in used_recipes}

# ----------------------------
# GUI
# ----------------------------
DEFAULT_RAW = [
    "Iron Ore (Limestone)",
    "Copper Ore (Quartz)",
    "Limestone (Sulfur)",
    "Coal (Iron)",
    "Sulfur (Coal)",
    "Raw Quartz (Bauxite)",
    "Caterium Ore (Copper)",
    "Bauxite (Caterium)",
    "SAM",
    "Water",
    "Liquid Oil",
    "Nitrogen Gas (Bauxite)",
]

class App(tk.Tk):
    def __init__(self, recipes_path):
        super().__init__()
        self.title("Satisfactory Clean Planner (LP-balanced)")
        self.geometry("980x640")

        # Load recipes
        with open(recipes_path, "r", encoding="utf-8") as f:
            self.all_recipes = json.load(f)

        # Build item list from recipe outputs
        items = set()
        for r in self.all_recipes.values():
            for o in (r.get("out") or {}):
                items.add(o)
        self.all_items_sorted = sorted(items)

        # UI vars
        self.target_var = tk.StringVar(value=self.all_items_sorted[0] if self.all_items_sorted else "")
        self.rate_var = tk.StringVar(value="60")  # items/min default
        self.filter_var = tk.BooleanVar(value=True)  # relevant-only by default

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Target item").grid(row=0, column=0, sticky="w")
        self.target_box = ttk.Combobox(top, textvariable=self.target_var, values=self.all_items_sorted, width=48)
        self.target_box.grid(row=1, column=0, sticky="w")

        ttk.Label(top, text="Target rate (items/min)").grid(row=0, column=1, sticky="w", padx=(16, 0))
        ttk.Entry(top, textvariable=self.rate_var, width=12).grid(row=1, column=1, sticky="w", padx=(16, 0))

        ttk.Checkbutton(
            top,
            text="Use only recipes relevant to target (recommended)",
            variable=self.filter_var
        ).grid(row=1, column=2, sticky="w", padx=(16, 0))

        # Raw items selector
        mid = ttk.Frame(self, padding=12)
        mid.pack(fill="x")

        ttk.Label(mid, text="Raw items you will import (Ctrl/Cmd+Click to multi-select)").grid(row=0, column=0, sticky="w")

        self.raw_list = tk.Listbox(mid, height=8, selectmode="extended", exportselection=False)
        self.raw_list.grid(row=1, column=0, sticky="we")
        mid.columnconfigure(0, weight=1)

        # Populate raw list with common options + everything seen in inputs
        all_inputs = set()
        for r in self.all_recipes.values():
            for i in (r.get("in") or {}):
                all_inputs.add(i)
        raw_candidates = sorted(set(DEFAULT_RAW) | all_inputs)

        for it in raw_candidates:
            self.raw_list.insert("end", it)

        # preselect DEFAULT_RAW if present
        for idx, it in enumerate(raw_candidates):
            if it in DEFAULT_RAW:
                self.raw_list.selection_set(idx)

        # Compute button
        ttk.Button(self, text="Compute clean plan", command=self.compute).pack(pady=(0, 8))

        # Output panes
        bottom = ttk.Panedwindow(self, orient="horizontal")
        bottom.pack(fill="both", expand=True, padx=12, pady=12)

        left = ttk.Frame(bottom, padding=8)
        right = ttk.Frame(bottom, padding=8)
        bottom.add(left, weight=1)
        bottom.add(right, weight=1)

        ttk.Label(left, text="Buildings (100% clock; fractional allowed)").pack(anchor="w")
        self.buildings_txt = tk.Text(left, wrap="none")
        self.buildings_txt.pack(fill="both", expand=True)

        ttk.Label(right, text="Net raw requirements (items/min) and notes").pack(anchor="w")
        self.raw_txt = tk.Text(right, wrap="none")
        self.raw_txt.pack(fill="both", expand=True)

    def compute(self):
        target = self.target_var.get().strip()
        if not target:
            messagebox.showerror("Missing target", "Please select a target item.")
            return

        try:
            rate = float(self.rate_var.get().strip())
        except Exception:
            messagebox.showerror("Bad rate", "Target rate must be a number (items/min).")
            return

        selected = [self.raw_list.get(i) for i in self.raw_list.curselection()]
        raw_items = set(selected)

        # Narrow recipe set for speed/cleanliness
        if self.filter_var.get():
            recipes = build_relevant_recipe_subset(self.all_recipes, target, raw_items)
        else:
            recipes = self.all_recipes

        if not recipes:
            messagebox.showerror(
                "No recipes found",
                "No recipes were found that produce the selected target (or you filtered them all out)."
            )
            return

        try:
            buildings, ratio, net = solve_satisfactory_clean_plan(
                recipes=recipes,
                target_item=target,
                target_rate_per_min=rate,
                raw_items=raw_items,
                allow_extra_outputs=None
            )
        except Exception as e:
            messagebox.showerror("Solve failed", str(e))
            return

        # Render buildings
        self.buildings_txt.delete("1.0", "end")
        lines = []
        for rname, b in sorted(buildings.items(), key=lambda kv: float(kv[1]), reverse=True):
            if b == 0:
                continue
            building = recipes[rname].get("building", "?")
            # show as fraction and approx
            approx = float(b)
            lines.append(f"{approx:10.4f} buildings  |  {building:14s}  |  {rname}  (exact {b})")
        if not lines:
            lines = ["(No buildings needed?) This usually means target rate is 0 or target is allowed as a byproduct."]
        self.buildings_txt.insert("end", "\n".join(lines))

        # Render raw requirements + sanity notes
        self.raw_txt.delete("1.0", "end")
        raw_lines = ["Raw inputs (net negative):"]
        any_raw = False
        for it, v in sorted(net.items(), key=lambda kv: float(kv[1])):
            if v < 0:
                any_raw = True
                raw_lines.append(f"  {it}: {float(-v):.4f} /min   (exact {-v})")
        if not any_raw:
            raw_lines.append("  (None)")

        raw_lines.append("\nByproducts (net positive, excluding target):")
        any_by = False
        for it, v in sorted(net.items(), key=lambda kv: float(kv[1]), reverse=True):
            if it != target and v > 0:
                any_by = True
                raw_lines.append(f"  {it}: {float(v):.4f} /min   (exact {v})")
        if not any_by:
            raw_lines.append("  (None)")

        raw_lines.append("\nInteger ratio (scale-up blueprint):")
        ratio_lines = []
        for rname, k in ratio.items():
            if k != 0:
                building = recipes[rname].get("building", "?")
                ratio_lines.append(f"  {k:6d}  |  {building:14s}  |  {rname}")
        raw_lines.extend(ratio_lines[:200])
        if len(ratio_lines) > 200:
            raw_lines.append(f"  ... ({len(ratio_lines)-200} more)")

        raw_lines.append("\nNotes:")
        raw_lines.append("- Recipes with empty inputs exist in the dataset (e.g., some Converter recipes).")
        raw_lines.append("  If you set such an item as 'allowed extra output', it can cause weird plans.")
        raw_lines.append("  Keep allow_extra_outputs=None for now unless you know what you're doing.")

        self.raw_txt.insert("end", "\n".join(raw_lines))

def main():
    # Path to your uploaded recipe file
    recipes_path = "/mnt/data/recipes_compact.json"
    app = App(recipes_path)
    app.mainloop()

if __name__ == "__main__":
    main()
