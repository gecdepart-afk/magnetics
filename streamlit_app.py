import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from discretize import TensorMesh
from simpeg import maps
from simpeg.potential_fields import magnetics
from simpeg.potential_fields.magnetics import receivers, sources, simulation


# ---------------------------
# Helpers: rotations
# ---------------------------
def Rz(deg: float) -> np.ndarray:
    """Rotation around +Z (degrees)."""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])

def Rx(deg: float) -> np.ndarray:
    """Rotation around +X (degrees)."""
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  ca, -sa],
                     [0.0,  sa,  ca]])


def build_dyke_mask(mesh: TensorMesh,
                    center_xyz=(0.0, 0.0, -45.0),
                    width=40.0,
                    length=500.0,
                    z_top=-10.0,
                    z_bottom=-80.0,
                    strike_deg=0.0,
                    dip_deg=90.0) -> np.ndarray:
    """
    Build a 'dyke-like' prism mask by defining a local coordinate system for the dyke:
      - u axis: across-dyke (controls width)
      - v axis: along-strike (controls length)
      - w axis: normal to dyke plane (thin direction) is not used; instead we bound by depth (z_top/z_bottom)

    We implement strike by rotating around Z, and dip by rotating around X AFTER strike rotation.
    This is a teaching-friendly approximation: you get a tilted/rotated dyke volume.
    """
    cc = mesh.cell_centers
    cx, cy, cz = center_xyz

    # Shift to dyke center
    p = cc - np.array([cx, cy, cz])

    # Define rotation: first align strike in map view, then dip it
    # Local coordinates = (R * p^T)^T
    # Choose strike rotation about Z, then dip about X (in the rotated frame)
    R = Rx(90.0 - dip_deg) @ Rz(strike_deg)
    ploc = (R @ p.T).T

    # In local coords:
    # ploc[:,0] ~ across-dyke (width)
    # ploc[:,1] ~ along-strike (length)
    # Depth is still controlled in global z for simplicity (robust for students)
    half_w = width / 2.0
    half_L = length / 2.0

    xloc = ploc[:, 0]
    yloc = ploc[:, 1]
    zglob = cc[:, 2]

    mask = (
        (np.abs(xloc) <= half_w) &
        (np.abs(yloc) <= half_L) &
        (zglob <= z_top) &
        (zglob >= z_bottom)
    )
    return mask


@st.cache_resource
def make_mesh(cs: float, nx: int, ny: int, nz: int) -> TensorMesh:
    hx = [(cs, nx)]
    hy = [(cs, ny)]
    hz = [(cs, nz)]
    return TensorMesh([hx, hy, hz], x0="CCC")


def main():
    st.set_page_config(page_title="SimPEG Magnetics – Dyke Demo", layout="wide")
    st.title("SimPEG Magnetics – Dyke forward modelling (interactive)")

    st.markdown(
        "This app models a dyke-like prism with **induced magnetization only** (susceptibility χ). "
        "You can rotate **the inducing field** (incl/decl) and rotate/tilt the **dyke geometry** (strike/dip)."
    )

    with st.sidebar:
        st.header("Mesh / performance")
        cs = st.slider("Cell size (m)", 10.0, 40.0, 20.0, 5.0)
        nx = st.slider("Nx cells", 20, 80, 40, 5)
        ny = st.slider("Ny cells", 20, 120, 60, 10)
        nz = st.slider("Nz cells", 15, 60, 30, 5)

        st.header("Dyke parameters")
        chi_dyke = st.slider("Dyke susceptibility χ (SI)", 0.0, 0.2, 0.05, 0.005)
        width = st.slider("Dyke width (m)", 10.0, 200.0, 40.0, 5.0)
        length = st.slider("Dyke length (m)", 100.0, 2000.0, 500.0, 50.0)
        z_top = st.slider("Top depth z_top (m, negative)", -5.0, -200.0, -10.0, 5.0)
        z_bottom = st.slider("Bottom depth z_bottom (m, negative)", -10.0, -400.0, -80.0, 10.0)

        strike = st.slider("Strike (deg)", 0.0, 360.0, 0.0, 5.0)
        dip = st.slider("Dip (deg)", 10.0, 90.0, 90.0, 5.0)

        st.header("Inducing field")
        B0 = st.slider("Field amplitude (nT)", 20000, 70000, 50000, 1000)
        inc = st.slider("Inclination (deg)", -90.0, 90.0, 60.0, 5.0)
        dec = st.slider("Declination (deg)", -180.0, 180.0, 0.0, 5.0)

        st.header("Survey")
        x_min, x_max = st.slider("Profile x-range (m)", -1000, 1000, (-200, 200), 50)
        n_pts = st.slider("Number of points", 51, 401, 161, 10)
        y_line = st.slider("Profile y (m)", -500, 500, 0, 10)
        z_obs = st.slider("Observation z (m)", -50, 50, 0, 1)

        plot_slice = st.checkbox("Plot susceptibility x–z slice", value=True)

        run = st.button("Run forward model")

    if not run:
        st.info("Set parameters in the sidebar, then click **Run forward model**.")
        return

    # ---------------------------
    # Build mesh + model
    # ---------------------------
    mesh = make_mesh(cs, nx, ny, nz)

    model = np.zeros(mesh.nC)
    active_cells = np.ones(mesh.nC, dtype=bool)

    # Dyke centered roughly in the middle
    center_xyz = (0.0, 0.0, 0.5 * (z_top + z_bottom))
    dyke_mask = build_dyke_mask(
        mesh,
        center_xyz=center_xyz,
        width=width,
        length=length,
        z_top=z_top,
        z_bottom=z_bottom,
        strike_deg=strike,
        dip_deg=dip,
    )
    model[dyke_mask] = chi_dyke

    # ---------------------------
    # Survey
    # ---------------------------
    x_profile = np.linspace(x_min, x_max, n_pts)
    y_profile = np.ones_like(x_profile) * float(y_line)
    z_profile = np.ones_like(x_profile) * float(z_obs)

    rx = receivers.Point(np.c_[x_profile, y_profile, z_profile], components=["tmi"])
    src_field = sources.UniformBackgroundField(
        receiver_list=[rx],
        amplitude=float(B0),
        inclination=float(inc),
        declination=float(dec),
    )
    survey = magnetics.Survey(src_field)

    # ---------------------------
    # Forward simulation
    # ---------------------------
    sim = simulation.Simulation3DIntegral(
        mesh=mesh,
        survey=survey,
        chiMap=maps.IdentityMap(nP=mesh.nC),
        active_cells=active_cells,
        store_sensitivities="ram",
    )

    tmi = sim.dpred(model)

    # ---------------------------
    # Plot results
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure()
        plt.plot(x_profile, tmi)
        plt.xlabel("x (m)")
        plt.ylabel("TMI anomaly (nT)")
        plt.title("TMI profile (induced magnetization, SimPEG)")
        plt.grid(True)
        st.pyplot(fig1)

    with col2:
        if plot_slice:
            # x–z slice at y ≈ y_line
            nx_c, ny_c, nz_c = mesh.shape_cells
            m3d = model.reshape((nx_c, ny_c, nz_c), order="F")

            y_centers = mesh.cell_centers_y
            iy = int(np.argmin(np.abs(y_centers - float(y_line))))

            x_centers = mesh.cell_centers_x
            z_centers = mesh.cell_centers_z

            slice_xz = m3d[:, iy, :].T

            fig2 = plt.figure()
            plt.imshow(
                slice_xz,
                extent=[x_centers.min(), x_centers.max(), z_centers.min(), z_centers.max()],
                origin="lower",
                aspect="auto",
            )
            plt.xlabel("x (m)")
            plt.ylabel("z (m)")
            plt.title(f"Susceptibility slice (x–z) at y ≈ {y_centers[iy]:.1f} m")
            plt.colorbar(label="χ (SI)")
            st.pyplot(fig2)
        else:
            st.write("Susceptibility slice disabled.")

    st.markdown(
        "**Teaching note:** In this app, the anomaly depends on **(1) field inclination/declination** and "
        "**(2) dyke strike/dip**, because magnetics is a **vector + geometry** problem, not just ± polarity."
    )


if __name__ == "__main__":
    main()
