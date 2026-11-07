import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# Find the XFOIL executable on Streamlit Cloud or a local computer.
_xfoil_location = shutil.which("xfoil")

if _xfoil_location:
    XFOIL_PATH = Path(_xfoil_location)
else:
    XFOIL_PATH = (
        Path.home()
        / "tools"
        / "xfoil-applesilicon"
        / "bin"
        / "xfoil"
    )


# XFOIL prints one or more aerodynamic solution lines in this format:
# a = 4.000 CL = 0.5367 Cm = -0.0149 CD = 0.01100
_SOLUTION_PATTERN = re.compile(
    r"a\s*=\s*([-+0-9.Ee]+)\s+"
    r"CL\s*=\s*([-+0-9.Ee]+)\s+"
    r"Cm\s*=\s*([-+0-9.Ee]+)\s+"
    r"CD\s*=\s*([-+0-9.Ee]+)",
    re.IGNORECASE,
)

_TRANSITION_PATTERN = re.compile(
    r"Side\s+([12]).*?transition at x/c\s*=\s*([-+0-9.Ee]+)",
    re.IGNORECASE,
)


def _parse_solver_output(output):
    """Extract aerodynamic solutions and transition locations from XFOIL output."""
    results = []
    transitions = {1: np.nan, 2: np.nan}

    event_pattern = re.compile(
        r"Side\s+([12]).*?transition at x/c\s*=\s*([-+0-9.Ee]+)"
        r"|a\s*=\s*([-+0-9.Ee]+)\s+"
        r"CL\s*=\s*([-+0-9.Ee]+)\s+"
        r"Cm\s*=\s*([-+0-9.Ee]+)\s+"
        r"CD\s*=\s*([-+0-9.Ee]+)",
        re.IGNORECASE | re.DOTALL,
    )

    for match in event_pattern.finditer(output):
        if match.group(1):
            side = int(match.group(1))
            transitions[side] = float(match.group(2))
            continue

        alpha, cl, cm, cd = map(float, match.groups()[2:])
        if not all(np.isfinite(v) for v in (alpha, cl, cm, cd)) or cd <= 0:
            continue

        row = {
            "alpha": alpha,
            "CL": cl,
            "CD": cd,
            "CDp": np.nan,
            "CM": cm,
            "Top_Xtr": transitions[1],
            "Bot_Xtr": transitions[2],
            "CL_CD": cl / cd,
        }

        # Replace the previous iteration for the same angle with the latest.
        if results and abs(results[-1]["alpha"] - alpha) < 1e-6:
            results[-1] = row
        else:
            results.append(row)

    return results


def _run_xfoil(naca, reynolds, mach, analysis_command):
    if not XFOIL_PATH.is_file():
        raise FileNotFoundError(
            f"XFOIL executable not found: {XFOIL_PATH}. "
            "Install a Linux-compatible XFOIL binary on the deployment server."
        )

    if not re.fullmatch(r"\d{4}", str(naca)):
        raise ValueError("NACA code must contain exactly four digits.")
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("Reynolds number must be greater than zero.")
    if not np.isfinite(mach) or mach < 0 or mach > 1:
        raise ValueError("Mach number must be between 0 and 1.")

    # Do not use PACC here. Reading the solution directly from stdout avoids
    # polar-file prompts and preserves valid results if XFOIL crashes on exit.
    commands = (
        "PLOP\n"
        "G\n"
        "\n"
        f"NACA {naca}\n"
        "PANE\n"
        "OPER\n"
        f"VISC {reynolds:.0f}\n"
        f"MACH {mach:.4f}\n"
        "ITER 200\n"
        f"{analysis_command}\n"
        "\n"
        "QUIT\n"
    )

    with tempfile.TemporaryDirectory(prefix="aerofoilx_") as temp_dir:
        try:
            process = subprocess.run(
                [str(XFOIL_PATH)],
                input=commands,
                text=True,
                capture_output=True,
                cwd=temp_dir,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial_output = exc.stdout or ""
            if isinstance(partial_output, bytes):
                partial_output = partial_output.decode(errors="replace")
            partial_results = _parse_solver_output(partial_output)
            if partial_results:
                return partial_results, partial_output
            raise RuntimeError("XFOIL timed out before producing aerodynamic data.") from exc

    output = (process.stdout or "") + "\n" + (process.stderr or "")
    results = _parse_solver_output(output)

    # XFOIL sometimes raises SIGFPE after printing a valid converged solution.
    # Keep valid results instead of rejecting them because of the exit code.
    if results:
        return results, output

    details = output[-5000:] if output else "XFOIL returned no terminal output."
    raise RuntimeError(
        "XFOIL did not print a valid converged aerodynamic solution. "
        f"Exit code: {process.returncode}\n{details}"
    )


def run_xfoil(naca, reynolds, alpha, mach=0.0):
    results, output = _run_xfoil(
        naca=naca,
        reynolds=reynolds,
        mach=mach,
        analysis_command=f"ALFA {float(alpha):.3f}",
    )
    result = results[-1].copy()
    result["solver_output"] = output
    return result


def run_xfoil_sweep(
    naca,
    reynolds,
    alpha_start=-5,
    alpha_end=15,
    alpha_step=1,
    mach=0.0,
):
    if alpha_step <= 0:
        raise ValueError("Alpha step must be positive.")
    if alpha_start >= alpha_end:
        raise ValueError("Starting angle must be smaller than ending angle.")

    command = (
        f"ASEQ {float(alpha_start):.3f} "
        f"{float(alpha_end):.3f} {float(alpha_step):.3f}"
    )

    results, _ = _run_xfoil(
        naca=naca,
        reynolds=reynolds,
        mach=mach,
        analysis_command=command,
    )
    return results


def run_xfoil_cp(naca, reynolds, alpha, mach=0.0):
    """Return XFOIL pressure-coefficient data, with an inviscid fallback."""
    if not XFOIL_PATH.is_file():
        raise FileNotFoundError(f"XFOIL executable not found: {XFOIL_PATH}")

    if not re.fullmatch(r"\d{4}", str(naca)):
        raise ValueError("NACA code must contain exactly four digits.")
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("Reynolds number must be greater than zero.")
    if not np.isfinite(mach) or mach < 0 or mach > 1:
        raise ValueError("Mach number must be between 0 and 1.")

    def read_cp_file(cp_file):
        if not cp_file.exists() or cp_file.stat().st_size == 0:
            return None
        try:
            data = np.loadtxt(cp_file, comments="#", ndmin=2)
        except (ValueError, OSError):
            return None
        if data.shape[1] < 2:
            return None
        data = data[np.all(np.isfinite(data[:, :2]), axis=1)]
        return data if data.shape[0] else None

    logs = []

    with tempfile.TemporaryDirectory(prefix="aerofoilx_cp_") as temp_dir:
        cp_file = Path(temp_dir) / "cp.dat"

        # Viscous attempt. No PLOP commands are needed for Cp output.
        viscous_commands = (
            f"NACA {naca}\n"
            "PANE\n"
            "OPER\n"
            f"VISC {reynolds:.0f}\n"
            f"MACH {mach:.4f}\n"
            "ITER 200\n"
            f"ALFA {float(alpha):.3f}\n"
            "CPWR cp.dat\n"
            "\n"
            "QUIT\n"
        )

        try:
            proc = subprocess.run(
                [str(XFOIL_PATH)],
                input=viscous_commands,
                text=True,
                capture_output=True,
                cwd=temp_dir,
                timeout=120,
                check=False,
            )
            logs.append((proc.stdout or "") + "\n" + (proc.stderr or ""))
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            logs.append(partial)

        data = read_cp_file(cp_file)
        cp_mode = "viscous" if data is not None else None

        # If the viscous solver crashes before CPWR, retry without VISC.
        if data is None:
            cp_file.unlink(missing_ok=True)
            inviscid_commands = (
                f"NACA {naca}\n"
                "PANE\n"
                "OPER\n"
                f"MACH {mach:.4f}\n"
                f"ALFA {float(alpha):.3f}\n"
                "CPWR cp.dat\n"
                "\n"
                "QUIT\n"
            )

            try:
                proc2 = subprocess.run(
                    [str(XFOIL_PATH)],
                    input=inviscid_commands,
                    text=True,
                    capture_output=True,
                    cwd=temp_dir,
                    timeout=120,
                    check=False,
                )
                logs.append((proc2.stdout or "") + "\n" + (proc2.stderr or ""))
            except subprocess.TimeoutExpired as exc:
                partial = exc.stdout or ""
                if isinstance(partial, bytes):
                    partial = partial.decode(errors="replace")
                logs.append(partial)

            data = read_cp_file(cp_file)
            if data is not None:
                cp_mode = "inviscid"

        if data is None:
            raise RuntimeError(
                "XFOIL failed to write pressure coefficient data in both "
                "viscous and inviscid modes.\n" + "\n".join(logs)[-5000:]
            )

        return {
            "x": data[:, 0],
            "Cp": data[:, 1],
            "solver_output": "\n".join(logs),
            "cp_mode": cp_mode,
        }
