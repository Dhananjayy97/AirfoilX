```python
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np


# ==================================================
# 1. FIND XFOIL EXECUTABLE
# ==================================================

def find_xfoil():

    # First, check system PATH
    xfoil_location = shutil.which("xfoil")

    if xfoil_location:
        return Path(xfoil_location)

    # Check local Mac installation
    mac_path = (
        Path.home()
        / "tools"
        / "xfoil-applesilicon"
        / "bin"
        / "xfoil"
    )

    if mac_path.is_file():
        return mac_path

    raise FileNotFoundError(
        "XFOIL executable not found. "
        "Install XFOIL and add it to the system PATH."
    )


XFOIL_PATH = find_xfoil()


# ==================================================
# 2. PARSE POLAR DATA
# ==================================================

def parse_polar(polar_file):

    results = []

    if not polar_file.exists():
        return results

    with open(polar_file, "r") as file:

        for line in file:

            values = line.split()

            if len(values) < 7:
                continue

            try:
                numbers = [float(value) for value in values[:7]]
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


# ==================================================
# 3. COMMON XFOIL EXECUTION FUNCTION
# ==================================================

def execute_xfoil(commands, timeout=120):

    if not XFOIL_PATH.is_file():
        raise FileNotFoundError(
            f"XFOIL executable not found: {XFOIL_PATH}"
        )

    result = subprocess.run(
        [str(XFOIL_PATH)],
        input=commands,
        text=True,
        capture_output=True,
        timeout=timeout
    )

    output = result.stdout + "\n" + result.stderr

    return output


# ==================================================
# 4. SINGLE POINT AND SWEEP ANALYSIS
# ==================================================

def _run_xfoil(naca, reynolds, mach, analysis_command):

    with tempfile.TemporaryDirectory(
        dir="/tmp",
        prefix="xf"
    ) as temp_dir:

        polar_file = Path(temp_dir) / "polar.txt"

        commands = f"""
PLOP
G
F

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

        output = execute_xfoil(commands)

        if not polar_file.exists():

            raise RuntimeError(
                "XFOIL did not generate a polar file.\n\n"
                + output[-4000:]
            )

        results = parse_polar(polar_file)

        if not results:

            raise RuntimeError(
                "No converged aerodynamic data found.\n\n"
                + output[-4000:]
            )

        return results, output


# ==================================================
# 5. SINGLE ANGLE ANALYSIS
# ==================================================

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


# ==================================================
# 6. ANGLE OF ATTACK SWEEP
# ==================================================

def run_xfoil_sweep(
    naca,
    reynolds,
    alpha_start=-5,
    alpha_end=15,
    alpha_step=1,
    mach=0.0
):

    if alpha_step <= 0:
        raise ValueError(
            "Alpha step must be positive."
        )

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


# ==================================================
# 7. PRESSURE COEFFICIENT ANALYSIS
# ==================================================

def run_xfoil_cp(
    naca,
    reynolds,
    alpha,
    mach=0.0
):

    with tempfile.TemporaryDirectory(
        dir="/tmp",
        prefix="xfcp"
    ) as temp_dir:

        cp_file = Path(temp_dir) / "pressure.txt"

        commands = f"""
PLOP
G
F

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

        output = execute_xfoil(commands)

        if not cp_file.exists():

            raise RuntimeError(
                "XFOIL did not generate pressure data.\n\n"
                + output[-4000:]
            )

        data = np.loadtxt(
            cp_file,
            comments="#",
            ndmin=2
        )

        if data.shape[0] == 0 or data.shape[1] < 2:

            raise RuntimeError(
                "Invalid pressure coefficient file.\n\n"
                + output[-4000:]
            )

        return {
            "x": data[:, 0],
            "Cp": data[:, 1],
            "solver_output": output
        }
```