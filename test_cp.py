
from solver.xfoil_runner import run_xfoil_cp

print("Starting pressure coefficient analysis...")

result = run_xfoil_cp(
    naca="2412",
    reynolds=1000000,
    alpha=4,
    mach=0.0
)

print("\nPRESSURE COEFFICIENT RESULTS")

print("Total data points:", len(result["x"]))

print("\nFirst 10 points:")

for x, cp in zip(result["x"][:10], result["Cp"][:10]):
    print(f"x/c = {x:.4f}, Cp = {cp:.4f}")

print("\nPressure coefficient extraction completed.")
