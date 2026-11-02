import plotly.express as px
import plotly.graph_objects as go

from solver.xfoil_runner import (
    run_xfoil_sweep,
    run_xfoil_cp
)
from solver.xfoil_runner import run_xfoil
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="AirfoilX",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AirfoilX")
st.subheader("Interactive Airfoil Analysis Platform")

st.write(
    "Generate airfoil geometry and calculate "
    "basic aerodynamic parameters."
)

# NACA 4-digit airfoil generator
def generate_airfoil(naca, points=150):

    m = int(naca[0]) / 100
    p = int(naca[1]) / 10
    t = int(naca[2:]) / 100

    # Cosine-spaced coordinates
    beta = np.linspace(0, np.pi, points)
    x = (1 - np.cos(beta)) / 2

    # Thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if m > 0 and p > 0:
        first = x < p
        second = ~first

        yc[first] = (
            m / p**2 *
            (2*p*x[first] - x[first]**2)
        )

        dyc[first] = (
            2*m / p**2 *
            (p - x[first])
        )

        yc[second] = (
            m / (1-p)**2 *
            (1 - 2*p + 2*p*x[second]
             - x[second]**2)
        )

        dyc[second] = (
            2*m / (1-p)**2 *
            (p - x[second])
        )

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


# Sidebar inputs
st.sidebar.header("Airfoil Parameters")

naca = st.sidebar.selectbox(
    "Select NACA airfoil",
    ["0012", "2412", "4412", "23012"]
)

velocity = st.sidebar.slider(
    "Flight velocity (m/s)", 5, 100, 25
)

chord = st.sidebar.number_input(
    "Chord length (m)", value=0.15, min_value=0.01
)

density = st.sidebar.number_input(
    "Air density (kg/m³)", value=1.225
)

viscosity = st.sidebar.number_input(
    "Dynamic viscosity (Pa.s)",
    value=1.81e-5,
    format="%.7f"
)

alpha = st.sidebar.slider(
    "Angle of Attack (°)",
    min_value=-5.0,
    max_value=15.0,
    value=4.0,
    step=0.5
)

mach = st.sidebar.number_input(
    "Mach Number",
    min_value=0.0,
    max_value=0.8,
    value=0.0,
    step=0.05
)



# Geometry
xu, yu, xl, yl = generate_airfoil(naca)

# Calculations
reynolds = density * velocity * chord / viscosity

dynamic_pressure = 0.5 * density * velocity**2

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Airfoil Geometry")

    fig, ax = plt.subplots()

    ax.plot(xu, yu, label="Upper Surface")
    ax.plot(xl, yl, label="Lower Surface")

    ax.set_xlabel("Chord Position (m)")
    ax.set_ylabel("Vertical Position (m)")
    ax.set_title(f"NACA {naca}")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Aerodynamic Parameters")

    st.metric(
        "Reynolds Number",
        f"{reynolds:,.0f}"
    )

    st.metric(
        "Dynamic Pressure",
        f"{dynamic_pressure:.2f} Pa"
    )

    st.metric(
        "Flight Velocity",
        f"{velocity} m/s"
    )

# Download coordinates
coordinates = pd.DataFrame({
    "x_upper": xu,
    "y_upper": yu,
    "x_lower": xl,
    "y_lower": yl
})

st.download_button(
    "Download Airfoil Coordinates",
    coordinates.to_csv(index=False),
    file_name=f"NACA_{naca}.csv",
    mime="text/csv"
)

st.caption(
    "AirfoilX | Academic Airfoil Design Tool"
)
# XFOIL Aerodynamic Analysis

st.divider()

st.header("🚀 XFOIL Aerodynamic Analysis")

st.write(
    "Run aerodynamic simulations using the XFOIL solver."
)

if "xfoil_result" not in st.session_state:
    st.session_state.xfoil_result = None

if st.button("Run XFOIL Analysis", type="primary"):

    try:
        with st.spinner("Running aerodynamic simulation..."):

            result = run_xfoil(
                naca=naca,
                reynolds=reynolds,
                alpha=alpha,
                mach=mach
            )

            st.session_state.xfoil_result = result

    except Exception as e:
        st.error(f"Simulation failed: {e}")

if st.session_state.xfoil_result is not None:

    result = st.session_state.xfoil_result

    st.subheader("Aerodynamic Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Lift Coefficient", f"{result['CL']:.4f}")
    c2.metric("Drag Coefficient", f"{result['CD']:.5f}")
    c3.metric("Moment Coefficient", f"{result['CM']:.4f}")
    c4.metric("Lift / Drag", f"{result['CL_CD']:.2f}")

    st.subheader("Boundary Layer Transition")

    t1, t2 = st.columns(2)

    t1.metric(
        "Upper Surface",
        f"{result['Top_Xtr'] * 100:.1f}%"
    )

    t2.metric(
        "Lower Surface",
        f"{result['Bot_Xtr'] * 100:.1f}%"
    )

    st.success("XFOIL simulation completed successfully!")

    
# ANGLE OF ATTACK SWEEP DASHBOARD

st.divider()

st.header("📊 Aerodynamic Performance Dashboard")

st.write(
    "Analyze airfoil performance across multiple "
    "angles of attack using XFOIL."
)

with st.expander("Sweep Settings", expanded=True):

    sweep_col1, sweep_col2, sweep_col3 = st.columns(3)

    with sweep_col1:
        alpha_start = st.number_input(
            "Starting AoA (°)", value=-5.0
        )

    with sweep_col2:
        alpha_end = st.number_input(
            "Ending AoA (°)", value=15.0
        )

    with sweep_col3:
        alpha_step = st.number_input(
            "AoA Step (°)",
            value=1.0,
            min_value=0.5
        )

if st.button("Run Aerodynamic Sweep"):

    try:
        with st.spinner("Running XFOIL sweep..."):

            sweep_results = run_xfoil_sweep(
                naca=naca,
                reynolds=reynolds,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                alpha_step=alpha_step,
                mach=mach
            )

            st.session_state.sweep_results = sweep_results

    except Exception as e:
        st.error(f"Sweep failed: {e}")


if "sweep_results" in st.session_state:

    df = pd.DataFrame(
        st.session_state.sweep_results
    )

    st.success(
        f"{len(df)} converged simulation points generated."
    )

    # Lift Curve
    st.subheader("1. Lift Curve")

    fig_lift = px.line(
        df,
        x="alpha",
        y="CL",
        markers=True,
        title="Lift Coefficient vs Angle of Attack",
        labels={
            "alpha": "Angle of Attack (°)",
            "CL": "Lift Coefficient"
        }
    )

    fig_lift.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_lift,
        use_container_width=True
    )

    # Drag Polar
    st.subheader("2. Drag Polar")

    fig_drag = px.scatter(
        df,
        x="CD",
        y="CL",
        color="alpha",
        hover_data=["CM", "CL_CD"],
        title="Drag Polar",
        labels={
            "CD": "Drag Coefficient",
            "CL": "Lift Coefficient",
            "alpha": "Angle of Attack (°)"
        }
    )

    fig_drag.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_drag,
        use_container_width=True
    )

    # Lift-to-Drag Ratio
    st.subheader("3. Aerodynamic Efficiency")

    fig_efficiency = px.line(
        df,
        x="alpha",
        y="CL_CD",
        markers=True,
        title="Lift-to-Drag Ratio vs Angle of Attack",
        labels={
            "alpha": "Angle of Attack (°)",
            "CL_CD": "CL/CD"
        }
    )

    fig_efficiency.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_efficiency,
        use_container_width=True
    )

    # Results Table
    st.subheader("Simulation Data")

    display_columns = [
        "alpha",
        "CL",
        "CD",
        "CM",
        "CL_CD"
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True
    )

    st.download_button(
        "Download Sweep Results CSV",
        df.to_csv(index=False),
        file_name=f"NACA_{naca}_sweep.csv",
        mime="text/csv"
    )

# PRESSURE COEFFICIENT ANALYSIS

st.divider()

st.header("🌬️ Pressure Coefficient Distribution")

st.write(
    "Analyze pressure variation across the upper "
    "and lower surfaces using XFOIL."
)

if st.button("Run Pressure Analysis"):

    try:
        with st.spinner("Calculating pressure distribution..."):

            cp_result = run_xfoil_cp(
                naca=naca,
                reynolds=reynolds,
                alpha=alpha,
                mach=mach
            )

            x = cp_result["x"]
            cp = cp_result["Cp"]

            # Locate leading edge
            leading_edge_index = int(x.argmin())

            # Separate upper and lower surfaces
            upper_x = x[:leading_edge_index + 1]
            upper_cp = cp[:leading_edge_index + 1]

            lower_x = x[leading_edge_index:]
            lower_cp = cp[leading_edge_index:]

            st.session_state.cp_data = {
                "upper_x": upper_x.tolist(),
                "upper_cp": upper_cp.tolist(),
                "lower_x": lower_x.tolist(),
                "lower_cp": lower_cp.tolist()
            }

    except Exception as e:
        st.error(f"Pressure analysis failed: {e}")


if "cp_data" in st.session_state:

    cp_data = st.session_state.cp_data

    fig_cp = go.Figure()

    fig_cp.add_trace(
        go.Scatter(
            x=cp_data["upper_x"],
            y=cp_data["upper_cp"],
            mode="lines+markers",
            name="Upper Surface"
        )
    )

    fig_cp.add_trace(
        go.Scatter(
            x=cp_data["lower_x"],
            y=cp_data["lower_cp"],
            mode="lines+markers",
            name="Lower Surface"
        )
    )

    fig_cp.update_layout(
        title=f"Pressure Coefficient Distribution, NACA {naca}",
        xaxis_title="Chord Position (x/c)",
        yaxis_title="Pressure Coefficient (Cp)",
        template="plotly_dark",
        height=550
    )

    fig_cp.update_yaxes(
        autorange="reversed"
    )

    fig_cp.update_xaxes(
        range=[0, 1]
    )

    st.plotly_chart(
        fig_cp,
        use_container_width=True
    )

    st.success(
        "Pressure coefficient analysis completed!"
    )

    cp_df = pd.DataFrame({
        "Upper_x": pd.Series(cp_data["upper_x"]),
        "Upper_Cp": pd.Series(cp_data["upper_cp"]),
        "Lower_x": pd.Series(cp_data["lower_x"]),
        "Lower_Cp": pd.Series(cp_data["lower_cp"])
    })

    st.download_button(
        "Download Cp Data CSV",
        cp_df.to_csv(index=False),
        file_name=f"NACA_{naca}_Cp_Data.csv",
        mime="text/csv"
    )

# ==========================================
# AIRFOIL COMPARISON MODULE
# ==========================================

st.divider()
st.header("Airfoil Comparison")
st.write("Compare aerodynamic performance of multiple NACA airfoils.")

comparison_airfoils = st.multiselect(
    "Select 2 to 4 airfoils",
    ["0012", "2412", "4412", "23012"],
    default=["0012", "2412"]
)

if st.button("Run Airfoil Comparison"):

    if len(comparison_airfoils) < 2:
        st.error("Please select at least 2 airfoils.")

    elif len(comparison_airfoils) > 4:
        st.error("Please select a maximum of 4 airfoils.")

    else:
        comparison_data = []

        progress = st.progress(0)
        status = st.empty()

        for index, code in enumerate(comparison_airfoils):

            status.write(f"Analysing NACA {code}...")

            try:
                results = run_xfoil_sweep(
                    naca=code,
                    reynolds=reynolds,
                    alpha_start=-5,
                    alpha_end=15,
                    alpha_step=1,
                    mach=mach
                )

                for result in results:
                    row = dict(result)
                    row["Airfoil"] = f"NACA {code}"
                    comparison_data.append(row)

            except Exception as error:
                st.error(f"NACA {code} failed: {error}")

            progress.progress((index + 1) / len(comparison_airfoils))

        status.empty()

        if comparison_data:
            st.session_state["comparison_data"] = pd.DataFrame(
                comparison_data
            )
            st.success("Airfoil comparison completed!")

if "comparison_data" in st.session_state:

    df_compare = st.session_state["comparison_data"]

    st.subheader("Aerodynamic Performance Comparison")

    fig_lift = px.line(
        df_compare,
        x="alpha",
        y="CL",
        color="Airfoil",
        markers=True,
        title="Lift Coefficient vs Angle of Attack"
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    fig_drag = px.line(
        df_compare,
        x="alpha",
        y="CD",
        color="Airfoil",
        markers=True,
        title="Drag Coefficient vs Angle of Attack"
    )
    st.plotly_chart(fig_drag, use_container_width=True)
    st.write("Available columns:", df_compare.columns.tolist())
   
    # Ensure lift-to-drag ratio exists
    df_compare["CL/CD"] = (
        df_compare["CL"] / df_compare["CD"]
    )

    fig_efficiency = px.line(
        df_compare,
        x="alpha",
        y="CL/CD",
        color="Airfoil",
        markers=True,
        title="Lift-to-Drag Ratio vs Angle of Attack"
    )

    st.plotly_chart(
        fig_efficiency,
        use_container_width=True
    )


    st.subheader("Comparison Data")
    st.dataframe(df_compare, use_container_width=True)

    csv_compare = df_compare.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Comparison CSV",
        data=csv_compare,
        file_name="aerofoilx_comparison.csv",
        mime="text/csv"
    )

# ==========================================
# STALL ANALYSIS MODULE
# ==========================================

st.divider()
st.header("Stall Analysis")
st.write(
    "Estimate stall behaviour using the angle-of-attack "
    "sweep results from XFOIL."
)

if "comparison_data" in st.session_state:

    stall_df = st.session_state["comparison_data"].copy()

    stall_df = stall_df.dropna(subset=["alpha", "CL", "CD"])

    stall_summary = []

    for airfoil in stall_df["Airfoil"].unique():

        airfoil_df = stall_df[
            stall_df["Airfoil"] == airfoil
        ].sort_values("alpha")

        if airfoil_df.empty:
            continue

        max_index = airfoil_df["CL"].idxmax()
        max_row = airfoil_df.loc[max_index]

        stall_summary.append({
            "Airfoil": airfoil,
            "Maximum CL": round(max_row["CL"], 4),
            "Estimated Stall Angle (°)": round(
                max_row["alpha"], 2
            ),
            "Drag at Maximum CL": round(
                max_row["CD"], 5
            ),
            "Maximum CL/CD": round(
                (airfoil_df["CL"] / airfoil_df["CD"]).max(),
                2
            )
        })

    if stall_summary:

        stall_results = pd.DataFrame(stall_summary)

        st.subheader("Stall Performance Summary")

        st.dataframe(
            stall_results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Maximum Lift Coefficient")

        fig_stall = px.bar(
            stall_results,
            x="Airfoil",
            y="Maximum CL",
            color="Airfoil",
            title="Maximum Lift Coefficient by Airfoil"
        )

        st.plotly_chart(
           fig_stall,
           use_container_width=True,
           key="stall_max_cl"
)
        

        st.subheader("Estimated Stall Angle")

        fig_angle = px.bar(
            stall_results,
            x="Airfoil",
            y="Estimated Stall Angle (°)",
            color="Airfoil",
            title="Estimated Stall Angle by Airfoil"
        )

        st.plotly_chart(
          fig_angle,
          use_container_width=True,
          key="stall_angle"
)

        st.subheader("Lift Behaviour Near Stall")

        fig_lift_stall = px.line(
            stall_df,
            x="alpha",
            y="CL",
            color="Airfoil",
            markers=True,
            title="Lift Coefficient vs Angle of Attack"
        )

        st.plotly_chart(
           fig_lift_stall,
           use_container_width=True,
           key="stall_lift_curve"
)

        st.download_button(
            "Download Stall Analysis CSV",
            data=stall_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="aerofoilx_stall_analysis.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Run Airfoil Comparison first to generate "
        "data for Stall Analysis."
    )
