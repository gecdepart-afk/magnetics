import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from discretize import TensorMesh
from simpeg import maps
from simpeg.potential_fields import magnetics
from simpeg.potential_fields.magnetics import receivers, sources, simulation


def build_mesh(cs=10.0, nx=40, ny=60, nz=30):
    hx = [(cs, nx)]
    hy = [(cs, ny)]
    hz = [(cs, nz)]
    mesh = TensorMesh([hx, hy, hz], x0="CCC")
    return mesh


def main():
    st.set_page_config(page_title="Magnetic dyke dip demo (SimPEG)", layout="wide")
    st.title("Magnetics forward modelling – dyke dip (simple 1D, fast)")

    st.markdown(
        "This demo uses **induced magnetization only** (susceptibility χ). "
        "You can change the **dyke dip** and see how the **1D TMI profile** changes. "
        "The right plot shows the dyke geometry (x–z slice)."
    )

    with st.sidebar:
        st.header("Mesh (speed vs resolution)")
        cs = st.slider("Cell size (m)", 10.0, 40.0, 10.0, 5.0)
        nx = st.slider("Nx", 20, 80, 40, 5)
        ny = st.slider("Ny", 20, 120, 60, 10)
        nz = st.slider("Nz", 15, 60, 30, 5)

        st.header("Dyke geometry")
        chi_dyke = st.slider("Susceptibility χ (SI)", 0.0, 0.2, 0.05, 0.005)
        half_width = st.slider("Half-width (m)", 5.0, 100.0, 20.0, 5.0)
        half_length = st.slider("Half-length along strike (m)", 50.0, 600.0, 250.0, 25.0)
        z_top = st.slider("Top depth z_top (m, negative)", -5.0, -200.0, -10.0, 5.0)
        z_bottom = st.slider("Bottom depth z_bottom (m, negative)", -10.0, -400.0, -80.0, 10.0)

        dip_deg = st.slider("Dyke dip (deg)", 30.0, 90.0, 90.0, 5.0)
        st.caption("Dip = 90° is vertical. Smaller dip = more inclined.")

        st.header("Inducing field")
        B0 = st.slider("Amplitude (nT)", 20000, 70000, 50000, 1000)
        inc = st.slider("Inclination (deg)", -90.0, 90.0, 60.0, 5.0)
        dec = st.slider("Declination (deg)", -180.0, 180.0, 0.0, 5.0)

        st.header("Profile")
        x_min, x_max = st.slider("Profile x-range (m)", -1000, 1000, (-200, 200), 50)
        n_pts = st.slider("Number of points", 51, 401, 161, 10)
        y_line = st.slider("Profile y (m)", -200, 200, 0, 10)
        z_obs = st.slider("Observation z (m)", -50, 50, 0, 1)

        run = st.button("Run forward model")

    if not run:
        st.info("Choose parameters in the sidebar, then click **Run forward model**.")
        return

    # ---------------------------
    # 1) Mesh
    # ---------------------------
    mesh = build_mesh(cs=float(cs), nx=int(nx), ny=int(ny), nz=int(nz))

    # ---------------------------
    # 2) Susceptibility model (induced magnetization only)
    # ---------------------------
    model = np.zeros(mesh.nC)
    active_cells = np.ones(mesh.nC, dtype=bool)

    x, y, z = mesh.cell_centers.T

    # --- Dipping dyke: a very simple, intuitive implementation ---
    # The dyke "center" shifts in x with depth, controlled by dip.
    # For a vertical dyke (dip=90°), x_shift ~ 0 at all depths.
    dip = np.deg2rad(float(dip_deg))
    dip = np.clip(dip, np.deg2rad(1.0), np.deg2rad(89.999)) if dip_deg < 90 else dip

    # Reference depth for the shear (use top of dyke)
    z0 = float(z_top)

    # Shift of dyke center with depth (in meters)
    # As z decreases (more negative), (z - z0) becomes negative.
    # Sign convention: this makes the dyke "lean" in +x for typical settings.
    if float(dip_deg) >= 89.999:
        x_shift = 0.0 * z
    else:
        x_shift = (z - z0) / np.tan(dip)

    dyke_mask = (
        (x >= (-float(half_width) + x_shift)) & (x <= (float(half_width) + x_shift)) &
        (y >= -float(half_length)) & (y <= float(half_length)) &
        (z <= float(z_top)) & (z >= float(z_bottom))
    )
    model[dyke_mask] = float(chi_dyke)

    # ---------------------------
    # 3) Survey: 1D profile along x at fixed y and z
    # ---------------------------
    x_profile = np.linspace(x_min, x_max, int(n_pts))
    y_profile = np.ones_like(x_profile) * float(y_line)
    z_profile = np.ones_like(x_profile) * float(z_obs)

    rx = receivers.Point(np.c_[x_profile, y_profile, z_profile], components=["tmi"])

    src_field = sources.UniformBackgroundField(
        receiver_list=[rx],
        amplitude=float(B0),          # nT
        inclination=float(inc),       # degrees
        declination=float(dec),       # degrees
    )
    survey = magnetics.Survey(src_field)

    # ---------------------------
    # 4) Forward simulation
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
    # 5) Plots
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure()
        plt.plot(x_profile, tmi)
        plt.xlabel("x (m)")
        plt.ylabel("TMI anomaly (nT)")
        plt.title("1D TMI profile")
        plt.grid(True)
        st.pyplot(fig1)

        st.caption(
            f"Dip = {dip_deg:.0f}°, χ = {chi_dyke:.3f} SI | "
            f"Field inc/dec = {inc:.0f}°/{dec:.0f}° | Profile y={y_line} m, z={z_obs} m"
        )

    with col2:
        # Plot x–z slice at y closest to the profile y (so dyke tilt is visible)
        nx_c, ny_c, nz_c = mesh.shape_cells
        m3d = model.reshape((nx_c, ny_c, nz_c), order="F")

        y_centers = mesh.cell_centers_y
        iy = int(np.argmin(np.abs(y_centers - float(y_line))))

        x_centers = mesh.cell_centers_x
        z_centers = mesh.cell_centers_z
        slice_xz = m3d[:, iy, :].T  # transpose so z is vertical

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

    st.markdown(
        "**Teaching tip:** Keep strike fixed and only vary dip (90° → 60° → 45°). "
        "Students will see the profile become asymmetric and shift because the body moves laterally with depth."
    )


if __name__ == "__main__":
    main()
