
from solver.xfoil_runner import run_xfoil_sweep

print("Starting XFOIL angle-of-attack sweep...")

results = run_xfoil_sweep(
    naca="2412",
    reynolds=1000000,
    alpha_start=-5,
    alpha_end=15,
    alpha_step=1,
    mach=0.0
)

print("\nAERODYNAMIC SWEEP RESULTS")

print(
    f"{'Alpha':>8} {'CL':>10} "
    f"{'CD':>10} {'CM':>10} {'CL/CD':>10}"
)

for result in results:
    print(
        f"{result['alpha']:>8.2f} "
        f"{result['CL']:>10.4f} "
        f"{result['CD']:>10.5f} "
        f"{result['CM']:>10.4f} "
        f"{result['CL_CD']:>10.2f}"
    )

print(f"\nTotal successful points: {len(results)}")
