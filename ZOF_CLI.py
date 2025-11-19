#!/usr/bin/env python3
"""
ZOF_CLI.py

Zero Of Functions (ZOF) Solver - Command Line Interface

Implements:
1. Bisection Method
2. Regula Falsi (False Position) Method
3. Secant Method
4. Newton-Raphson Method
5. Fixed Point Iteration Method
6. Modified Secant Method

Usage:
    python ZOF_CLI.py

This script runs an interactive menu. For functions or g(x) or derivative,
enter a Python expression in terms of x (use math module functions).
Examples:
    x**3 - x - 2
    math.cos(x) - x
    math.exp(-x) - x

Notes:
- Newton uses numeric derivative by default; you can optionally provide
  a derivative expression.
- Fixed Point requires g(x) (the iteration function x = g(x)).
- Modified Secant uses a small delta perturbation (you provide delta).
"""

import math
import sys
from typing import Callable, List, Tuple, Optional

# ---------------------------
# Utilities
# ---------------------------
def make_function(expr: str) -> Callable[[float], float]:
    """Return a function f(x) for the given expression string using math namespace."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    def f(x):
        try:
            return eval(expr, {"__builtins__": {}}, {**allowed, "x": x})
        except Exception as e:
            raise ValueError(f"Error evaluating expression at x={x}: {e}")
    return f

def numeric_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2*h)

def format_iter_row(i:int, x:float, fx:Optional[float], extra:Optional[dict]=None) -> str:
    row = f"{i:4d} | {x: .12e}"
    if fx is not None:
        row += f" | {fx: .12e}"
    else:
        row += " | " + " " * 15
    if extra:
        for v in extra.values():
            row += f" | {v: .12e}"
    return row


# Root-finding methods
# Each returns (root, iterations, converged_flag)
# iterations: list of dicts with iteration data


def bisection(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa = f(a); fb = f(b)
    if fa*fb > 0:
        raise ValueError("Bisection requires f(a) and f(b) to have opposite signs.")
    iterations = []
    for i in range(1, max_iter+1):
        c = 0.5*(a+b)
        fc = f(c)
        err = abs(b - a)/2
        iterations.append({"iter": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": err})
        if abs(fc) == 0 or err < tol:
            return c, iterations, True
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return c, iterations, False

def regula_falsi(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int):
    fa = f(a); fb = f(b)
    if fa*fb > 0:
        raise ValueError("Regula Falsi requires f(a) and f(b) to have opposite signs.")
    iterations = []
    c = a
    for i in range(1, max_iter+1):
        c_prev = c
        c = (a*fb - b*fa) / (fb - fa)  # false position formula
        fc = f(c)
        err = abs(c - c_prev) if i>1 else float('inf')
        iterations.append({"iter": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": err})
        if abs(fc) == 0 or err < tol:
            return c, iterations, True
        if fa*fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return c, iterations, False

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float, max_iter: int):
    iterations = []
    f0 = f(x0); f1 = f(x1)
    for i in range(1, max_iter+1):
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Zero division in secant method (f1 - f0 == 0).")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        iterations.append({"iter": i, "x0": x0, "x1": x1, "x2": x2, "f(x2)": f2, "error": err})
        if abs(f2) == 0 or err < tol:
            return x2, iterations, True
        x0, f0 = x1, f1
        x1, f1 = x2, f2
    return x2, iterations, False

def newton_raphson(f: Callable[[float], float], x0: float, tol: float, max_iter: int,
                   df: Optional[Callable[[float], float]] = None):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        if df:
            dfx = df(x)
        else:
            dfx = numeric_derivative(f, x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero. Newton-Raphson fails.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        iterations.append({"iter": i, "x": x, "f(x)": fx, "f'(x)": dfx, "x_new": x_new, "error": err})
        if abs(fx) == 0 or err < tol:
            return x_new, iterations, True
        x = x_new
    return x, iterations, False

def fixed_point_iteration(g: Callable[[float], float], x0: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        fx = None
        iterations.append({"iter": i, "x": x, "g(x)": x_new, "error": err})
        if err < tol:
            return x_new, iterations, True
        x = x_new
    return x, iterations, False

def modified_secant(f: Callable[[float], float], x0: float, delta: float, tol: float, max_iter: int):
    iterations = []
    x = x0
    for i in range(1, max_iter+1):
        f_x = f(x)
        denom = f(x + delta*x) - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero division in modified secant (denominator == 0).")
        x_new = x - (delta*x*f_x) / denom
        err = abs(x_new - x)
        iterations.append({"iter": i, "x": x, "f(x)": f_x, "x_new": x_new, "error": err})
        if abs(f_x) == 0 or err < tol:
            return x_new, iterations, True
        x = x_new
    return x, iterations, False

# ---------------------------
# Pretty printing of iterations
# ---------------------------

def print_bisection_iters(iters):
    print("Iter |      a (left)    |      b (right)   |      c (mid)    |     f(c)         |    error")
    print("-"*90)
    for d in iters:
        print(f"{d['iter']:4d} | {d['a']: .12e} | {d['b']: .12e} | {d['c']: .12e} | {d['f(c)']: .12e} | {d['error']: .12e}")

def print_regula_iters(iters):
    print_bisection_iters(iters)  # same columns

def print_secant_iters(iters):
    print("Iter |      x0           |      x1           |      x2           |    f(x2)         |    error")
    print("-"*100)
    for d in iters:
        print(f"{d['iter']:4d} | {d['x0']: .12e} | {d['x1']: .12e} | {d['x2']: .12e} | {d['f(x2)']: .12e} | {d['error']: .12e}")

def print_newton_iters(iters):
    print("Iter |      x           |     f(x)          |    f'(x)          |     x_new         |   error")
    print("-"*110)
    for d in iters:
        print(
    f"{d['iter']:4d} | "
    f"{d['x']: .12e} | "
    f"{d['f(x)']: .12e} | "
    f"{d['f_prime(x)']: .12e} | "
    f"{d['x_new']: .12e} | "
    f"{d['error']: .12e}"
)


def print_newton_iters_safe(iters):
    print("Iter |      x           |     f(x)          |    f'(x)          |     x_new         |   error")
    print("-"*110)
    for d in iters:
        fx = d.get("f(x)", float('nan'))
        dfx = d.get("f'(x)", float('nan'))
        x_new = d.get("x_new", float('nan'))
        print(f"{d['iter']:4d} | {d['x']: .12e} | {fx: .12e} | {dfx: .12e} | {x_new: .12e} | {d['error']: .12e}")

def print_fixed_iters(iters):
    print("Iter |      x           |     g(x)           |    error")
    print("-"*80)
    for d in iters:
        print(f"{d['iter']:4d} | {d['x']: .12e} | {d['g(x)']: .12e} | {d['error']: .12e}")

def print_modified_secant(iters):
    print("Iter |      x           |     f(x)           |    x_new          |    error")
    print("-"*100)
    for d in iters:
        print(f"{d['iter']:4d} | {d['x']: .12e} | {d['f(x)']: .12e} | {d['x_new']: .12e} | {d['error']: .12e}")

# ---------------------------
# Interactive CLI
# ---------------------------

def prompt_float(prompt_text: str, default: Optional[float] = None) -> float:
    while True:
        s = input(f"{prompt_text}" + (f" [{default}]" if default is not None else "") + ": ")
        if s.strip() == "" and default is not None:
            return default
        try:
            return float(s)
        except:
            print("Please enter a valid number.")

def prompt_int(prompt_text: str, default: Optional[int] = None) -> int:
    while True:
        s = input(f"{prompt_text}" + (f" [{default}]" if default is not None else "") + ": ")
        if s.strip() == "" and default is not None:
            return default
        try:
            return int(s)
        except:
            print("Please enter a valid integer.")

def main_menu():
    print("\nZOF - Zero Of Functions Solver (CLI)")
    print("Choose a method:")
    print("1) Bisection Method")
    print("2) Regula Falsi (False Position) Method")
    print("3) Secant Method")
    print("4) Newton-Raphson Method")
    print("5) Fixed Point Iteration Method")
    print("6) Modified Secant Method")
    print("0) Exit")
    choice = input("Enter choice (0-6): ").strip()
    return choice

def run():
    while True:
        choice = main_menu()
        if choice == "0":
            print("Goodbye.")
            sys.exit(0)

        if choice not in {"1","2","3","4","5","6"}:
            print("Invalid selection.")
            continue

        expr = input("Enter f(x) (example: x**3 - x - 2). Use math.<func> for functions (or just 'x' math builtins available):\n f(x) = ").strip()
        try:
            f = make_function(expr)
            # quick test eval
            _ = f(0.0)
        except Exception as e:
            print(f"Error in function expression: {e}")
            continue

        tol = prompt_float("Enter tolerance (e.g. 1e-6)", 1e-6)
        max_iter = prompt_int("Enter maximum iterations", 50)

        try:
            if choice == "1":  # Bisection
                a = prompt_float("Enter left endpoint a")
                b = prompt_float("Enter right endpoint b")
                root, iters, conv = bisection(f, a, b, tol, max_iter)
                print("\n--- Iterations ---")
                print_bisection_iters(iters)
                print("\nResult:")
                print(f"Estimated root: {root:.12e}")
                print(f"Final error: {iters[-1]['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

            elif choice == "2":  # Regula Falsi
                a = prompt_float("Enter left endpoint a")
                b = prompt_float("Enter right endpoint b")
                root, iters, conv = regula_falsi(f, a, b, tol, max_iter)
                print("\n--- Iterations ---")
                print_regula_iters(iters)
                print("\nResult:")
                print(f"Estimated root: {root:.12e}")
                print(f"Final error: {iters[-1]['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

            elif choice == "3":  # Secant
                x0 = prompt_float("Enter initial guess x0")
                x1 = prompt_float("Enter initial guess x1")
                root, iters, conv = secant(f, x0, x1, tol, max_iter)
                print("\n--- Iterations ---")
                print_secant_iters(iters)
                print("\nResult:")
                print(f"Estimated root: {root:.12e}")
                print(f"Final error: {iters[-1]['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

            elif choice == "4":  # Newton-Raphson
                x0 = prompt_float("Enter initial guess x0")
                use_df = input("Do you want to supply derivative f'(x)? (y/N): ").strip().lower()
                if use_df == "y":
                    dexpr = input("Enter f'(x) expression (in terms of x): ").strip()
                    try:
                        df = make_function(dexpr)
                        _ = df(x0)
                    except Exception as e:
                        print(f"Error in derivative expression: {e}")
                        continue
                else:
                    df = None
                root, iters, conv = newton_raphson(f, x0, tol, max_iter, df)
                print("\n--- Iterations ---")
                print_newton_iters_safe(iters)
                print("\nResult:")
                last = iters[-1]
                print(f"Estimated root: {root:.12e}")
                print(f"Final error: {last['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

            elif choice == "5":  # Fixed Point
                gexpr = input("Enter g(x) for fixed-point iteration (x = g(x)):\n g(x) = ").strip()
                try:
                    g = make_function(gexpr)
                    _ = g(0.0)
                except Exception as e:
                    print(f"Error in g(x) expression: {e}")
                    continue
                x0 = prompt_float("Enter initial guess x0")
                root, iters, conv = fixed_point_iteration(g, x0, tol, max_iter)
                print("\n--- Iterations ---")
                print_fixed_iters(iters)
                print("\nResult:")
                print(f"Estimated fixed point: {root:.12e}")
                print(f"Final error: {iters[-1]['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

            elif choice == "6":  # Modified Secant
                x0 = prompt_float("Enter initial guess x0")
                delta = prompt_float("Enter delta (e.g. 1e-3) (used as fractional perturbation)", 1e-3)
                root, iters, conv = modified_secant(f, x0, delta, tol, max_iter)
                print("\n--- Iterations ---")
                print_modified_secant(iters)
                print("\nResult:")
                print(f"Estimated root: {root:.12e}")
                print(f"Final error: {iters[-1]['error']:.12e}")
                print(f"Number of iterations: {len(iters)}")
                print(f"Converged: {conv}")

        except Exception as e:
            print(f"An error occurred during computation: {e}")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    run()
