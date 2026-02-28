from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from .camera import get_camera_view
from .model_types import CameraFrame, CameraPose, Field, PathPolyline, Pose3D

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pv = None


@dataclass
class FreeRoamCameraController:
    enabled: bool = False


class SimulationViewer:
    def __init__(
        self,
        field: Field,
        path: Optional[PathPolyline] = None,
        primary_camera: Optional[CameraPose] = None,
        use_pyvista: Optional[bool] = None,
    ):
        self.field = field
        self.path = path
        self.primary_camera = primary_camera or CameraPose(pose=Pose3D(0.0, 0.0, 1.5))
        self.free_roam = FreeRoamCameraController(enabled=False)
        # Default off for portability and to avoid X/display issues in headless environments.
        self._use_pyvista = bool(use_pyvista and pv)
        self._plotter = None

        if self._use_pyvista:
            # Off-screen avoids hard dependency on a local GUI for automated tests.
            self._plotter = pv.Plotter(off_screen=True)
            self._build_scene()

    def _build_scene(self) -> None:
        if not self._plotter:
            return

        self._plotter.clear()
        for gate in self.field.gates:
            cube = pv.Cube(
                center=(gate.pose.x, gate.pose.y, gate.pose.z),
                x_length=max(gate.config.width_m, 0.1),
                y_length=max(gate.config.frame_thickness_m, 0.05),
                z_length=max(gate.config.height_m, 0.1),
            )
            self._plotter.add_mesh(cube, color=gate.config.color, opacity=0.65)

        if self.path and self.path.points:
            path_lines = pv.lines_from_points(self.path.points, close=False)
            self._plotter.add_mesh(path_lines, color="yellow", line_width=3)

        bounds_min = self.field.config.bounds_min
        bounds_max = self.field.config.bounds_max
        outline = pv.Box(bounds=(
            bounds_min[0], bounds_max[0],
            bounds_min[1], bounds_max[1],
            bounds_min[2], bounds_max[2],
        ))
        self._plotter.add_mesh(outline, style="wireframe", color="white", opacity=0.2)
        self._plotter.show_grid()

    def set_free_roam(self, enabled: bool = True) -> None:
        self.free_roam.enabled = enabled

    def update_primary_camera(self, pose: CameraPose) -> None:
        self.primary_camera = pose

    def draw_path(self, path: PathPolyline) -> None:
        self.path = path
        if self._use_pyvista:
            self._build_scene()

    def snapshot(self, include_depth: bool = False) -> CameraFrame:
        return get_camera_view(self.field, self.primary_camera, include_depth=include_depth)

    def launch_free_roam(self, window_size: tuple[int, int] = (1280, 720)) -> None:
        # Wayland-only sessions often lack DISPLAY; many VTK builds still require X11.
        # Fall back to a matplotlib 3D interactive viewer in that case.
        if _should_use_matplotlib_fallback():
            self._launch_matplotlib_free_roam(window_size)
            return

        if pv is None:
            self._launch_matplotlib_free_roam(window_size)
            return
        if os.name != "nt" and not _has_graphical_session():
            raise RuntimeError(
                "Interactive free-roam requires a graphical display, but no X11/Wayland session was detected. "
                "Set DISPLAY (X11) or WAYLAND_DISPLAY (Wayland), use SSH X-forwarding (`ssh -X`), "
                "or run without --interactive."
            )

        plotter = pv.Plotter(off_screen=False, window_size=window_size)
        self._plotter = plotter
        self._use_pyvista = True
        self._build_scene()
        self.set_free_roam(True)

        # Start from primary camera pose, then allow user-driven free roam.
        cam = plotter.camera
        p = self.primary_camera.pose
        cam.position = (p.x, p.y, p.z)
        cam.focal_point = (p.x + 1.0, p.y, p.z)
        cam.up = (0.0, 0.0, 1.0)

        def _print_camera_pose() -> None:
            c = plotter.camera
            cx, cy, cz = c.position
            fx, fy, fz = c.focal_point
            print("camera.position =", (round(cx, 3), round(cy, 3), round(cz, 3)))
            print("camera.focal_point =", (round(fx, 3), round(fy, 3), round(fz, 3)))

        plotter.add_text(
            "Free roam: mouse to move, W/S zoom, press 'p' to print camera pose, 'q' to quit",
            position="upper_left",
            font_size=10,
        )
        plotter.add_key_event("p", _print_camera_pose)
        plotter.show()

    def _launch_matplotlib_free_roam(self, window_size: tuple[int, int]) -> None:
        try:
            import matplotlib as mpl
        except Exception as exc:
            raise RuntimeError(
                "Interactive free-roam fallback requires matplotlib. "
                "Run: bash scripts/setup_venv.sh"
            ) from exc

        if _is_noninteractive_mpl_backend(mpl.get_backend()):
            for candidate in ("QtAgg", "TkAgg"):
                try:
                    mpl.use(candidate, force=True)
                    break
                except Exception:
                    continue

        if _is_noninteractive_mpl_backend(mpl.get_backend()):
            raise RuntimeError(
                "Matplotlib fallback is using a non-interactive backend "
                f"({mpl.get_backend()!r}), so no window can open. "
                "Install a GUI backend (Qt or Tk), or run snapshot mode without --interactive."
            )

        import matplotlib.pyplot as plt
        import numpy as np

        width_inches = max(6.0, window_size[0] / 100.0)
        height_inches = max(4.0, window_size[1] / 100.0)
        fig = plt.figure(figsize=(width_inches, height_inches))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax_cam = fig.add_subplot(gs[0, 1])

        for gate in self.field.gates:
            ax.scatter(gate.pose.x, gate.pose.y, gate.pose.z, s=80, c=gate.config.color, marker="s")
            ax.text(gate.pose.x, gate.pose.y, gate.pose.z, gate.gate_id, fontsize=8)

        if self.path and self.path.points:
            xs = [p[0] for p in self.path.points]
            ys = [p[1] for p in self.path.points]
            zs = [p[2] for p in self.path.points]
            ax.plot(xs, ys, zs, color="yellow", linewidth=2.0)

        bmin = self.field.config.bounds_min
        bmax = self.field.config.bounds_max
        ax.set_xlim(bmin[0], bmax[0])
        ax.set_ylim(bmin[1], bmax[1])
        ax.set_zlim(bmin[2], bmax[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Free-Roam Map")

        pose = self.primary_camera.pose
        camera_marker = ax.scatter([pose.x], [pose.y], [pose.z], s=90, c="cyan", marker="o")
        heading_quiver = ax.quiver(
            pose.x,
            pose.y,
            pose.z,
            math.cos(pose.yaw),
            math.sin(pose.yaw),
            0.0,
            length=2.0,
            color="cyan",
        )

        frame = get_camera_view(self.field, self.primary_camera, include_depth=False)
        image_artist = ax_cam.imshow(np.array(frame.rgb, dtype=np.uint8))
        ax_cam.set_title("Drone Camera View")
        ax_cam.axis("off")

        move_step = 0.6
        yaw_step = 0.12
        pitch_step = 0.08
        z_step = 0.4

        def _apply_movement(key: str) -> bool:
            nonlocal pose
            changed = True
            x, y, z = pose.x, pose.y, pose.z
            yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

            fwd_x = math.cos(yaw)
            fwd_y = math.sin(yaw)
            right_x = -math.sin(yaw)
            right_y = math.cos(yaw)

            if key == "w":
                x += move_step * fwd_x
                y += move_step * fwd_y
            elif key == "s":
                x -= move_step * fwd_x
                y -= move_step * fwd_y
            elif key == "a":
                x -= move_step * right_x
                y -= move_step * right_y
            elif key == "d":
                x += move_step * right_x
                y += move_step * right_y
            elif key == "r":
                z += z_step
            elif key == "f":
                z -= z_step
            elif key == "j":
                yaw += yaw_step
            elif key == "l":
                yaw -= yaw_step
            elif key == "i":
                pitch = min(pitch + pitch_step, 1.2)
            elif key == "k":
                pitch = max(pitch - pitch_step, -1.2)
            elif key == "p":
                print(
                    "camera.pose =",
                    {
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "z": round(z, 3),
                        "yaw": round(yaw, 3),
                        "pitch": round(pitch, 3),
                        "roll": round(roll, 3),
                    },
                )
                changed = False
            else:
                return False

            if changed:
                pose = Pose3D(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
                self.primary_camera = CameraPose(
                    pose=pose,
                    fov_horizontal_deg=self.primary_camera.fov_horizontal_deg,
                    resolution_width=self.primary_camera.resolution_width,
                    resolution_height=self.primary_camera.resolution_height,
                    near_plane_m=self.primary_camera.near_plane_m,
                    far_plane_m=self.primary_camera.far_plane_m,
                )
            return True

        def _refresh_visuals() -> None:
            nonlocal camera_marker, heading_quiver
            frame_local = get_camera_view(self.field, self.primary_camera, include_depth=False)
            image_artist.set_data(np.array(frame_local.rgb, dtype=np.uint8))

            camera_marker.remove()
            heading_quiver.remove()
            camera_marker = ax.scatter([pose.x], [pose.y], [pose.z], s=90, c="cyan", marker="o")
            heading_quiver = ax.quiver(
                pose.x,
                pose.y,
                pose.z,
                math.cos(pose.yaw),
                math.sin(pose.yaw),
                0.0,
                length=2.0,
                color="cyan",
            )
            fig.canvas.draw_idle()

        def _on_key(event) -> None:
            if not event.key:
                return
            key = event.key.lower()
            if key == "q":
                plt.close(fig)
                return
            changed = _apply_movement(key)
            if changed:
                _refresh_visuals()

        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.suptitle(
            "Free-roam controls: W/S forward-back, A/D strafe, R/F up-down, J/L yaw, I/K pitch, P print pose, Q quit",
            fontsize=9,
        )
        self.set_free_roam(True)
        plt.tight_layout()
        plt.show()


def render_scene(
    field: Field,
    path: Optional[PathPolyline] = None,
    primary_camera: Optional[CameraPose] = None,
) -> SimulationViewer:
    return SimulationViewer(field=field, path=path, primary_camera=primary_camera)


def _has_graphical_session() -> bool:
    # X11
    if os.environ.get("DISPLAY"):
        return True
    # Native Wayland
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Some environments expose only session type.
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return True
    return False


def _should_use_matplotlib_fallback() -> bool:
    # If this is a Wayland session without X11 DISPLAY, PyVista/VTK commonly cannot open a window.
    if os.name == "nt":
        return False
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
        return True
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and not os.environ.get("DISPLAY"):
        return True
    return False


def _is_noninteractive_mpl_backend(backend_name: str) -> bool:
    name = backend_name.lower().strip()
    # Interactive backends we explicitly allow.
    if name in {"qtagg", "qt5agg", "tkagg", "gtk3agg", "gtk4agg", "macosx", "wxagg", "nbagg", "webagg"}:
        return False
    # Common non-interactive backends.
    if name in {"agg", "pdf", "svg", "ps", "template", "cairo"}:
        return True
    # Inline/notebook module backends are non-windowed for desktop roaming.
    if name.startswith("module://") and ("inline" in name or "ipympl" in name):
        return True
    return False
