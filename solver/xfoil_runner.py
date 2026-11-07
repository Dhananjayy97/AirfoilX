import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Find XFOIL automatically
xfoil_location = shutil.which("xfoil")

if xfoil_location:
    XFOIL_PATH = Path(xfoil_location)
else:
    XFOIL_PATH = (
        Path.home()
        / "tools"
        / "xfoil-applesilicon"
        / "bin"
        / "xfoil"
    )


def parse_polar(polar_file):
    results = []

    with open(polar_file, "r") as file:
        for line in file:
            values = line.split()

            if len(values) < 7:
                continue

            try:
                numbers = [
                    float(value) for value in values[:7]
                ]
            except ValueError:
                continue

            alpha, cl, cd, cdp, cm, top_xtr, bot_xtr = numbers

            results.append({
                "alpha": alpha,
                "CL": cl,
                "CD": cd,
                "CDp": cdp,
                "CM": cm,
                "Top_Xtr": top_xtr,
                "Bot_Xtr": bot_xtr,
                "CL_CD": cl / cd if cd != 0 else None
            })

    return results


def _run_xfoil(naca, reynolds, mach, analysis_command):

    if not XFOIL_PATH.exists():
        raise FileNotFoundError(
            f"XFOIL executable not found: {XFOIL_PATH}"
        )

    with tempfile.TemporaryDirectory(
        dir="/tmp",
        prefix="xf"
    ) as temp_dir:

        polar_file = Path(temp_dir) / "polar.txt"

        commands = f"""
NACA {naca}
PANE
OPER
VISC {reynolds:.0f}
MACH {mach}
ITER 200
PACC
{polar_file}

{analysis_command}
PACC

QUIT
"""

        result = subprocess.run(
    [str(XFOIL_PATH), "/dev/null"],
            input=commands,
            text=True,
            capture_output=True,
            timeout=120
        )

        if not polar_file.exists():
            raise RuntimeError(
                "XFOIL did not generate a polar file.\n"
                + result.stdout[-3000:]
                + result.stderr
            )

        results = parse_polar(polar_file)

        if not results:
            raise RuntimeError(
                "No converged aerodynamic data found.\n"
                + result.stdout[-3000:]
            )

        return results, result.stdout


def run_xfoil(naca, reynolds, alpha, mach=0.0):

    results, output = _run_xfoil(
        naca,
        reynolds,
        mach,
        f"ALFA {alpha}"
    )

    result = results[-1]
    result["solver_output"] = output

    return result


def run_xfoil_sweep(
    naca,
    reynolds,
    alpha_start=-5,
    alpha_end=15,
    alpha_step=1,
    mach=0.0
):

    if alpha_step <= 0:
        raise ValueError("Alpha step must be positive.")

    if alpha_start >= alpha_end:
        raise ValueError(
            "Starting angle must be smaller than ending angle."
        )

    command = (
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}"
    )

    results, output = _run_xfoil(
        naca,
        reynolds,
        mach,
        command
    )

    return results


def run_xfoil_cp(
    naca,
    reynolds,
    alpha,
    mach=0.0
):

    if not XFOIL_PATH.exists():
        raise FileNotFoundError(
            f"XFOIL executable not found: {XFOIL_PATH}"
        )

    with tempfile.TemporaryDirectory(
        dir="/tmp",
        prefix="xfcp"
    ) as temp_dir:

        cp_file = Path(temp_dir) / "pressure.txt"

        commands = f"""
NACA {naca}
PANE
OPER
VISC {reynolds:.0f}
MACH {mach}
ITER 200
ALFA {alpha}
CPWR {cp_file}
QUIT
"""

        result = subprocess.run(
    [str(XFOIL_PATH), "/dev/null"],
            input=commands,
            text=True,
            capture_output=True,
            timeout=120
        )

        if not cp_file.exists():
            raise RuntimeError(
                "XFOIL did not generate pressure data.\n"
                + result.stdout[-3000:]
                + result.stderr
            )

        data = np.loadtxt(cp_file, comments="#")

        if data.ndim != 2 or data.shape[1] < 2:
            raise RuntimeError(
                "Invalid pressure coefficient file."
            )
        return {
            "x": data[:, 0],
            "Cp": data[:, 1],
            "solver_output": result.stdout
        }
