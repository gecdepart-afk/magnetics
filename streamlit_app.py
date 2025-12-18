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
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])

def Rx(deg: float) -> np.ndarray:
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
    Dyke defined in a local frame (across-strike, along-strike) then rotated back.
    Depth bounds are applied in global z (simple + robust for teaching).
    """
    cc = mesh.cell_centers
    cx, cy, cz = center_xyz
    p = cc - np.array([cx, cy, cz])

    # strike rotation about Z, then dip (tilt) about X
    R = Rx(90.0 - dip_deg) @ Rz(strike_deg)
    ploc = (R @ p.T).T

    half_w = width / 2.0
    half_L = length / 2.0

    xloc = ploc[:, 0]     # across-dyke
    yloc = ploc[:, 1]     # along-strike
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


def make_profile(line_length: float, n_pts: int,
                 azimuth_deg: float, offset_m: float,
                 z_obs: float) -> np.ndarray:
    """
    Build a 1D profile line in x-y:
      - azimuth_deg: 0 = along +x, 90 = along +y (mathematical convention)
      - offset_m: shift perpendicular to the profile direction
    """
    t = np.linspace(-line_length / 2.0, line_length / 2.0, n_pts)

    a = np.deg2rad(azimuth_deg)
    u = np.array([np.cos(a), np.sin(a)])          # along-profile unit vector
    v = np.array([-np.sin(a), np.cos(a)])         # left-perp unit vector

    xy = np.outer(t, u) + offset_m * v[None, :]
    x = xy[:, 0]
    y = xy[:, 1]
    z = np.ones_like(x) * float(z_obs)

    return np.c_[x, y, z], t


def main():
    st.set_page_config(page_title="Fast 1D Dyke Magnetics (SimPEG)", layout="wide")
    st.title("Fast 1D magnetic forward modelling – dyke strike/dip vs profile direction")

    st.markdown(
        "This is **induced magnetization only** (susceptibility χ). "
        "To clearly see dyke rotation in **1D**, change the **profile azimuth** and **offset**."
    )

    with st.sidebar:
        st.header("Mesh (keep moderate for speed)")
        cs = st.slider("Cell size (m)", 10.0, 40.0, 20.0, 5.0)
        nx = st.slider("Nx", 20, 80, 40, 5)
        ny = st.slider("Ny", 20, 120, 60, 10)
        nz = st.slider("Nz", 15, 60, 30, 5)

        st.header("Dyke")
        chi_dyke = st.slider("χ dyke (SI)", 0.0, 0.2, 0.05, 0.005)
        width = st.slider("width (m)", 10.0, 200.0, 40.0, 5.0)
        length = st.slider("length (m)", 100.0, 1500.0, 400.0, 50.0)  # shorter helps see strike effects
        z_top = st.slider("z_top (m, negative)", -5.0, -200.0, -10.0, 5.0)
        z_bottom = st.slider("z_bottom (m, negative)", -10.0, -400.0, -80.0, 10.0)

        strike = st.slider("strike (deg)", 0.0, 360.0, 0.0, 5.0)
        dip = st.slider("dip (deg)", 10.0, 90.0, 90.0, 5.0)

        st.header("Inducing field")
        B0 = st.slider("B0 (nT)", 20000, 70000, 50000, 1000)
        inc = st.slider("inclination (deg)", -90.0, 90.0, 60.0, 5.0)
        dec = st.slider("declination (deg)", -180.0, 180.0, 0.0, 5.0)

        st.header("1D Profile (this is the key)")
        line_length = st.slider("profile length (m)", 200.0, 2000.0, 600.0, 50.0)
        n_pts = st.slider("number of points", 51, 301, 161, 10)
        prof_az = st.slider("profile azimuth (deg)", 0.0, 180.0, 0.0, 5.0)
        offset = st.slider("profile offset (m)", -300.0, 300.0, 120.0, 10.0)
        z_obs = st.slider("observation z (m)", -50.0, 50.0, 0.0, 1.0)

        show_slice = st.checkbox("Show x–z susceptibility slice (fast)", value=True)

        run = st.button("Run")

    if not run:
        st.info("Adjust parameters and click **Run**.")
        st.caption("Tip: set dyke strike = 90° and profile azimuth = 0° (or vice versa) and offset ≠ 0 to see changes clearly.")
        return

    # Mesh
    mesh = make_mesh(cs, nx, ny, nz)

    # Model
    model = np.zeros(mesh.nC)
    active_cells = np.ones(mesh.nC, dtype=bool)

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

    # 1D profile locations
    locs, t = make_profile(line_length, n_pts, prof_az, offset, z_obs)
    rx = receivers.Point(locs, components=["tmi"])

    src_field = sources.UniformBackgroundField(
        receiver_list=[rx],
        amplitude=float(B0),
        inclination=float(inc),
        declination=float(dec),
    )
    survey = magnetics.Survey(src_field)

    # Forward
    sim = simulation.Simulation3DIntegral(
        mesh=mesh,
        survey=survey,
        chiMap=maps.IdentityMap(nP=mesh.nC),
        active_cells=active_cells,
        store_sensitivities="ram",
    )
    tmi = sim.dpred(model)

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        fig = plt.figure()
        plt.plot(t, tmi)
        plt.xlabel("Distance along profile (m)")
        plt.ylabel("TMI anomaly (nT)")
        plt.title("1D TMI profile")
        plt.grid(True)
        st.pyplot(fig)

        st.caption(
            f"Profile azimuth = {prof_az:.0f}°, offset = {offset:.0f} m | "
            f"Dyke strike = {strike:.0f}°, dip = {dip:.0f}°"
        )

    with col2:
        if show_slice:
            nx_c, ny_c, nz_c = mesh.shape_cells
            m3d = model.reshape((nx_c, ny_c, nz_c), order="F")

            # slice at y ≈ 0 just for geometry visibility
            y_centers = mesh.cell_centers_y
            iy0 = int(np.argmin(np.abs(y_centers - 0.0)))

            x_centers = mesh.cell_centers_x
            z_centers = mesh.cell_centers_z
            slice_xz = m3d[:, iy0, :].T

            fig2 = plt.figure()
            plt.imshow(
                slice_xz,
                extent=[x_centers.min(), x_centers.max(), z_centers.min(), z_centers.max()],
                origin="lower",
                aspect="auto",
            )
            plt.xlabel("x (m)")
            plt.ylabel("z (m)")
            plt.title("Susceptibility slice (x–z) at y ≈ 0 (geometry check)")
            plt.colorbar(label="χ (SI)")
            st.pyplot(fig2)
        else:
            st.write("Slice disabled.")

    st.success(
        "If strike changes still look subtle: increase |offset|, shorten dyke length, "
        "or set profile azimuth very different from dyke strike (e.g., 0° vs 90°)."
    )


if __name__ == "__main__":
    main()
