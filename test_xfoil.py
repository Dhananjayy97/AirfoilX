
from solver.xfoil_runner import run_xfoil

result = run_xfoil(
    naca="2412",
    reynolds=1000000,
    alpha=4,
    mach=0.0
)
print("\nAERODYNAMIC ANALYSIS RESULTS")

for key, value in result.items():

    if key != "solver_output":
        print(f"{key}: {value}")
