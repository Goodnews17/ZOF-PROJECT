from flask import Flask, render_template, request
import math

from ZOF_CLI import bisection, secant, newton_raphson, regula_falsi, fixed_point_iteration, modified_secant

app = Flask(__name__)

# ---------------------------
# Home Page
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None
    iterations_table = None

    if request.method == "POST":
        method = request.form.get("method")
        equation = request.form.get("equation")
        x0 = float(request.form.get("x0"))
        x1 = request.form.get("x1")
        tol = float(request.form.get("tolerance"))
        max_iter = int(request.form.get("max_iter"))

        # Convert x1 only if provided (for methods that need two guesses)
        if x1 not in ["", None]:
            x1 = float(x1)

        # Convert string equation to a Python function f(x)
        def f(x):
            return eval(equation, {"x": x, "math": math})

        try:
            # ---------------------------
            # Method Routing
            # ---------------------------

            if method == "bisection":
                root, table,status = bisection(f, x0, x1, tol, max_iter)
                result = root
                iterations_table = table

            elif method == "regula_falsi":
                root, table,status = regula_falsi(f, x0, x1, tol, max_iter)
                result = root
                iterations_table = table

            elif method == "secant":
                root, table,status = secant(f, x0, x1, tol, max_iter)
                result = root
                iterations_table = table

            elif method == "newton":
                # Newton only needs x0
                root, table,status = newton_raphson(f, x0, tol, max_iter)
                result = root
                iterations_table = table

            elif method == "fixed_point":
                root, table,status = fixed_point_iteration(f, x0, tol, max_iter)
                result = root
                iterations_table = table

            elif method == "modified_secant":
                root, table,status = modified_secant(f, x0, tol, max_iter)
                result = root
                iterations_table = table

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template("index.html",
                           result=result,
                           error_message=error_message,
                           iterations_table=iterations_table)

# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
