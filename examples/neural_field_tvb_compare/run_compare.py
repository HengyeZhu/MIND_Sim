from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mind_sim as ms


@dataclass(frozen=True)
class TvbSurfaceData:
    connectivity: object
    surface: object
    labels: list[str]
    region_mapping: np.ndarray
    node_map: ms.NodeToRoiMap
    local: ms.LocalConnectivity
    local_csr: object
    roi_weights: np.ndarray
    delays: np.ndarray
    initial: np.ndarray
    initial_w: np.ndarray
    spatial_mean: np.ndarray
    vertices: int
    local_edges: int
    roi_edges: int


def add_tvb_root(tvb_root: Path) -> str:
    library = tvb_root / "tvb_library"
    if library.exists():
        sys.path.insert(0, str(library))
    try:
        import tvb  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"TVB import unavailable: {type(exc).__name__}: {exc}"
    status = f"TVB source: {Path(tvb.__file__).resolve()}"
    try:
        from tvb.simulator import monitors  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional TVB dependencies
        status += f"; tvb.simulator.monitors unavailable: {type(exc).__name__}: {exc}"
    else:
        status += "; tvb.simulator.monitors available"
    return status


def tvb_spatial_mean(region_mapping: np.ndarray, roi_count: int, surface: object) -> np.ndarray:
    from tvb.simulator import monitors

    class DummyIntegrator:
        dt = 1.0

    class DummyModel:
        variables_of_interest = ("x",)

    class DummySimulator:
        integrator = DummyIntegrator()
        model = DummyModel()
        number_of_nodes = int(region_mapping.size)

        def __init__(self, surface: object) -> None:
            self.surface = surface

    monitor = monitors.SpatialAverage(period=1.0)
    monitor.config_for_sim(DummySimulator(surface))
    spatial_mean = np.asarray(monitor.spatial_mean, dtype=np.float64)
    if spatial_mean.shape != (roi_count, region_mapping.size):
        raise RuntimeError(
            "TVB SpatialAverage produced unexpected shape "
            f"{spatial_mean.shape}; expected {(roi_count, region_mapping.size)}"
        )
    return spatial_mean


def deterministic_surface_initial(vertices: np.ndarray) -> np.ndarray:
    centered = vertices.astype(np.float64, copy=False) - np.mean(vertices, axis=0)
    radius = np.linalg.norm(centered, axis=1)
    scale = max(1e-12, float(np.max(radius)))
    x = centered[:, 0] / scale
    y = centered[:, 1] / scale
    z = centered[:, 2] / scale
    return 0.2 + 0.04 * x - 0.03 * y + 0.02 * z


def load_tvb_default_surface_data(
    *,
    tvb_root: Path,
    local_strength: float,
    global_strength: float,
    conduction_speed: float,
    use_delays: bool,
) -> tuple[str, TvbSurfaceData]:
    tvb_status = add_tvb_root(tvb_root)
    from tvb.datatypes import connectivity, cortex

    conn = connectivity.Connectivity.from_file()
    conn.speed = np.array([float(conduction_speed) if use_delays else np.inf], dtype=np.float64)
    conn.configure()

    surface = cortex.Cortex.from_file(local_connectivity_file="local_connectivity_16384.mat")
    surface.region_mapping_data.connectivity = conn
    surface.region_mapping_data.surface.configure()
    surface.coupling_strength = np.array([float(local_strength)], dtype=np.float64)
    surface.configure()

    region_mapping = np.asarray(surface.region_mapping, dtype=np.int32)
    roi_count = int(conn.number_of_regions)
    if np.unique(region_mapping).size != roi_count:
        raise RuntimeError("TVB default surface mapping does not cover every ROI")
    if int(region_mapping.min()) != 0 or int(region_mapping.max()) != roi_count - 1:
        raise RuntimeError("TVB default surface mapping is not contiguous from zero")

    local_csr = surface.prepare_local_coupling(region_mapping.size).tocsr()
    local_csr.sort_indices()
    local_csr.sum_duplicates()
    local = ms.LocalConnectivity.from_csr(local_csr)

    labels = [str(label) for label in np.asarray(conn.region_labels)]
    roi_weights = np.asarray(conn.weights, dtype=np.float64) * float(global_strength)
    delays = np.asarray(conn.delays, dtype=np.float64)
    delays = np.where(np.isfinite(delays), delays, 0.0)
    delays = np.maximum(delays, 0.0)

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    initial = deterministic_surface_initial(vertices)
    centered = vertices - np.mean(vertices, axis=0)
    radius = np.linalg.norm(centered, axis=1)
    scale = max(1e-12, float(np.max(radius)))
    initial_w = -0.1 + 0.01 * centered[:, 2] / scale
    spatial_mean = tvb_spatial_mean(region_mapping, roi_count, surface)

    data = TvbSurfaceData(
        connectivity=conn,
        surface=surface,
        labels=labels,
        region_mapping=region_mapping,
        node_map=ms.NodeToRoiMap.from_surface(surface),
        local=local,
        local_csr=local_csr,
        roi_weights=roi_weights,
        delays=delays,
        initial=initial,
        initial_w=initial_w,
        spatial_mean=spatial_mean,
        vertices=int(region_mapping.size),
        local_edges=local.nnz,
        roi_edges=int(np.count_nonzero(roi_weights)),
    )
    return tvb_status, data


def run_mind_linear(
    *,
    data: TvbSurfaceData,
    gamma: float,
    dt: float,
    t_stop: float,
) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory() as tmp:
        mod_dir = Path(tmp)
        (mod_dir / "field_drive.mod").write_text(
            """
MIND {
  COUPLING field_drive
  READ x
  WRITE drive
}

EDGE {
  drive += weight * x;
}
""",
            encoding="utf-8",
        )
        field_owner = """
  x = x + dt * (gamma * x + drive + local(x));
"""
        network = ms.Network(
            labels=data.labels,
            weights=data.roi_weights.tolist(),
            delays=data.delays.tolist(),
        )
        field = ms.NeuralField(
            "tvb_default_cortex",
            field_owner,
            inputs={"drive": 0.0},
            exposures="x",
            local=data.local,
            state={"x": data.initial.tolist()},
            params={"gamma": float(gamma)},
        )
        network.use_neural_field(field, node_map=data.node_map)
        roi_count = len(data.labels)
        for target in range(roi_count):
            for source in range(roi_count):
                if data.roi_weights[target, source] != 0.0:
                    network.roi(target).connect(network.roi(source), mod_dir / "field_drive.mod")

        simulator = ms.MacroSimulator(network, dt_macro=dt)
        start = time.perf_counter()
        result = simulator.run(t_stop)
    elapsed = time.perf_counter() - start
    width = result.exposures.recorded_roi_count * result.exposures.exposure_count
    final = np.asarray(result.exposures.values[-width:], dtype=np.float64)
    return final, elapsed


def run_mind_generic2d(
    *,
    data: TvbSurfaceData,
    dt: float,
    t_stop: float,
) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory() as tmp:
        mod_dir = Path(tmp)
        (mod_dir / "linear_drive.mod").write_text(
            """
MIND {
  COUPLING linear_drive
  READ V
  WRITE c_0
}

EDGE {
  c_0 += weight * V;
}
""",
            encoding="utf-8",
        )
        field_owner = """
  v2 = V * V;
  dV = d * tau * (alpha * W - f * v2 * V + e * v2 + g * V + gamma * I + gamma * c_0 + local(V));
  dW = d * (a + b * V + c * v2 - beta * W) / tau;
  V = V + dt * dV;
  W = W + dt * dW;
"""
        network = ms.Network(
            labels=data.labels,
            weights=data.roi_weights.tolist(),
            delays=data.delays.tolist(),
        )
        field = ms.NeuralField(
            "tvb_generic2d_cortex",
            field_owner,
            inputs={"c_0": 0.0},
            exposures="V",
            local=data.local,
            state={"V": data.initial.tolist(), "W": data.initial_w.tolist()},
            params={
                "tau": 1.0,
                "I": 0.0,
                "a": -2.0,
                "b": -10.0,
                "c": 0.0,
                "d": 0.02,
                "e": 3.0,
                "f": 1.0,
                "g": 0.0,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
            },
        )
        network.use_neural_field(field, node_map=data.node_map)
        roi_count = len(data.labels)
        for target in range(roi_count):
            for source in range(roi_count):
                if data.roi_weights[target, source] != 0.0:
                    network.roi(target).connect(network.roi(source), mod_dir / "linear_drive.mod")

        simulator = ms.MacroSimulator(network, dt_macro=dt)
        start = time.perf_counter()
        result = simulator.run(t_stop)
    elapsed = time.perf_counter() - start
    width = result.exposures.recorded_roi_count * result.exposures.exposure_count
    final = np.asarray(result.exposures.values[-width:], dtype=np.float64)
    return final, elapsed


def run_tvb_data_reference(
    *,
    data: TvbSurfaceData,
    gamma: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, float]:
    roi_count = len(data.labels)
    x = data.initial.astype(np.float64, copy=True)
    roi = data.spatial_mean @ x
    delay_steps = np.rint(data.delays / float(dt)).astype(np.int64)
    history_capacity = max(1, int(delay_steps.max()) + 1)
    history = np.tile(roi[np.newaxis, :], (history_capacity, 1))
    sources = np.arange(roi_count, dtype=np.int64)

    start = time.perf_counter()
    for step in range(steps):
        drive = np.zeros(roi_count, dtype=np.float64)
        current_slot = step % history_capacity
        for target in range(roi_count):
            slots = (current_slot - delay_steps[target]) % history_capacity
            delayed = history[slots, sources]
            drive[target] = float(np.dot(data.roi_weights[target], delayed))
        local = data.local_csr @ x
        x = x + dt * (gamma * x + drive[data.region_mapping] + local)
        roi = data.spatial_mean @ x
        history[(step + 1) % history_capacity, :] = roi
    elapsed = time.perf_counter() - start
    return roi, elapsed


def run_tvb_generic2d_simulator(
    *,
    data: TvbSurfaceData,
    dt: float,
    t_stop: float,
    global_strength: float,
) -> tuple[np.ndarray, float]:
    from tvb.simulator import coupling, integrators, models, simulator

    delay_steps = np.rint(data.delays / float(dt)).astype(np.int64)
    history_capacity = max(1, int(delay_steps.max()) + 1)
    initial_conditions = np.zeros((history_capacity, 2, data.vertices, 1), dtype=np.float64)
    initial_conditions[:, 0, :, 0] = data.initial
    initial_conditions[:, 1, :, 0] = data.initial_w

    sim = simulator.Simulator(
        connectivity=data.connectivity,
        conduction_speed=float(data.connectivity.speed[0]),
        coupling=coupling.Linear(a=np.array([float(global_strength)], dtype=np.float64)),
        surface=data.surface,
        integrator=integrators.EulerDeterministic(dt=float(dt)),
        simulation_length=float(t_stop),
        model=models.Generic2dOscillator(variables_of_interest=("V",)),
        monitors=(),
        initial_conditions=initial_conditions,
    )
    sim.configure()
    start = time.perf_counter()
    sim.run(simulation_length=float(t_stop))
    elapsed = time.perf_counter() - start
    final = data.spatial_mean @ np.asarray(sim.current_state[0, :, 0], dtype=np.float64)
    return final, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvb-root", type=Path, default=Path.home() / "tvb-root")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=-0.8)
    parser.add_argument("--local-strength", type=float, default=1.0)
    parser.add_argument("--global-strength", type=float, default=2.0**-8)
    parser.add_argument("--conduction-speed", type=float, default=3.0)
    parser.add_argument("--no-delays", action="store_true")
    args = parser.parse_args()

    data_status, data = load_tvb_default_surface_data(
        tvb_root=args.tvb_root,
        local_strength=float(args.local_strength),
        global_strength=float(args.global_strength),
        conduction_speed=float(args.conduction_speed),
        use_delays=not bool(args.no_delays),
    )
    t_stop = float(args.steps) * float(args.dt)
    mind_final, mind_seconds = run_mind_linear(
        data=data,
        gamma=float(args.gamma),
        dt=float(args.dt),
        t_stop=t_stop,
    )
    tvb_final, tvb_seconds = run_tvb_data_reference(
        data=data,
        gamma=float(args.gamma),
        dt=float(args.dt),
        steps=int(args.steps),
    )
    max_abs = float(np.max(np.abs(mind_final - tvb_final)))
    rel = max_abs / max(1e-15, float(np.max(np.abs(tvb_final))))

    mind_g2d_final, mind_g2d_seconds = run_mind_generic2d(
        data=data,
        dt=float(args.dt),
        t_stop=t_stop,
    )
    tvb_g2d_final, tvb_g2d_seconds = run_tvb_generic2d_simulator(
        data=data,
        dt=float(args.dt),
        t_stop=t_stop,
        global_strength=float(args.global_strength),
    )
    g2d_max_abs = float(np.max(np.abs(mind_g2d_final - tvb_g2d_final)))
    g2d_rel = g2d_max_abs / max(1e-15, float(np.max(np.abs(tvb_g2d_final))))

    print(data_status)
    print("data=TVB defaults: connectivity_76, cortex_16384, regionMapping_16k_76, local_connectivity_16384")
    print(
        f"surface_vertices={data.vertices} rois={len(data.labels)} "
        f"local_edges={data.local_edges} roi_edges={data.roi_edges}"
    )
    print(
        f"steps={args.steps} dt={args.dt} gamma={args.gamma} "
        f"local_strength={args.local_strength} global_strength={args.global_strength}"
    )
    print(
        f"delays={'off' if args.no_delays else 'on'} "
        f"conduction_speed={args.conduction_speed}"
    )
    print("linear_field_exact_reference:")
    print(f"  mind_sim_seconds={mind_seconds:.6f}")
    print(f"  tvb_data_reference_seconds={tvb_seconds:.6f}")
    print(f"  speedup_vs_reference={tvb_seconds / mind_seconds:.3f}")
    print(f"  max_abs_error={max_abs:.12e}")
    print(f"  max_rel_error={rel:.12e}")
    print("tvb_surface_simulator_generic2d:")
    print(f"  mind_sim_seconds={mind_g2d_seconds:.6f}")
    print(f"  tvb_simulator_seconds={tvb_g2d_seconds:.6f}")
    print(f"  speedup_vs_tvb_simulator={tvb_g2d_seconds / mind_g2d_seconds:.3f}")
    print(f"  max_abs_error={g2d_max_abs:.12e}")
    print(f"  max_rel_error={g2d_rel:.12e}")


if __name__ == "__main__":
    main()
