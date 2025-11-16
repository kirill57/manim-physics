"""Simple harmonic oscillators built on top of :mod:`rigid_mechanics`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from manim.constants import LEFT, RIGHT, UP
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import Rectangle
from manim.mobject.types.vectorized_mobject import VMobject, VGroup
import numpy as np
import pymunk

from .rigid_mechanics import SpaceScene

__all__ = ["SpringBlockOscillator"]


@dataclass
class SpringStyle:
    """Parameters that define the appearance of the spring."""

    coils: int = 12
    amplitude: float = 0.1
    stroke_width: float = 4
    color: Optional[str] = None


class SpringBlockOscillator(VGroup):
    """A block connected to a fixed wall with a spring."""

    def __init__(
        self,
        rest_length: float = 2.5,
        stiffness: float = 18,
        damping: float = 0.7,
        block_mass: float = 1.5,
        block_size: tuple[float, float] = (0.8, 0.8),
        anchor_point=np.array([-4.0, 0.0, 0.0]),
        spring_style: SpringStyle | None = None,
        block_style: dict | None = None,
        anchor_style: dict | None = None,
        animate_spring: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping
        self.block_mass = block_mass
        self.block_width, self.block_height = block_size
        self.animate_spring = animate_spring
        self._spacescene: Optional[SpaceScene] = None
        self.spring_joint: Optional[pymunk.DampedSpring] = None

        self._spring_style = spring_style or SpringStyle()
        self._anchor_style = anchor_style or {}
        self._block_style = block_style or {}

        self.anchor = Line(
            anchor_point + 0.6 * UP,
            anchor_point + 0.6 * DOWN,
            **self._anchor_style,
        )

        self.block = Rectangle(
            width=self.block_width,
            height=self.block_height,
            **self._block_style,
        )
        self.block.move_to(anchor_point + RIGHT * (rest_length + self.block_width / 2))

        self.spring = VMobject(
            stroke_width=self._spring_style.stroke_width,
            color=self._spring_style.color,
        )
        self._redraw_spring()

        self.add(self.anchor, self.spring, self.block)

        if self.animate_spring:
            self.spring.add_updater(lambda mob: self._redraw_spring())

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _get_anchor_point(self) -> np.ndarray:
        return self.anchor.point_from_proportion(0.5)

    def _block_local_anchor(self) -> pymunk.Vec2d:
        return pymunk.Vec2d(-self.block_width / 2, 0)

    def _block_anchor_point(self) -> np.ndarray:
        if hasattr(self.block, "body"):
            offset = self._block_local_anchor().rotated(self.block.body.angle)
            pos = self.block.body.position + offset
            return np.array([pos.x, pos.y, 0.0])
        return self.block.get_center() + LEFT * self.block_width / 2

    def _redraw_spring(self) -> None:
        anchor = self._get_anchor_point()
        block_point = self._block_anchor_point()
        axis = block_point - anchor
        length = np.linalg.norm(axis[:2])
        if length == 0:
            axis = RIGHT
            length = 1
        direction = axis / length
        normal = np.array([-direction[1], direction[0], 0])

        points = []
        resolution = max(4 * self._spring_style.coils, 8)
        ts = np.linspace(0, 1, resolution)
        for t in ts:
            sine = np.sin(2 * np.pi * self._spring_style.coils * t)
            points.append(
                anchor + axis * t + normal * sine * self._spring_style.amplitude
            )
        self.spring.set_points_smoothly(points)

    # ------------------------------------------------------------------
    # Pymunk helpers
    # ------------------------------------------------------------------
    def attach_to_scene(self, scene: SpaceScene) -> None:
        """Register the oscillator with a :class:`SpaceScene`."""

        self._spacescene = scene
        scene.make_static_body(self.anchor)

        block_area = self.block_width * self.block_height
        density = self.block_mass / block_area if block_area else 1
        scene.make_rigid_body(self.block, density=density)

        self._create_joint()

    def _create_joint(self) -> None:
        if not self._spacescene or not hasattr(self.block, "body"):
            return
        if self.spring_joint is not None:
            return

        anchor_xy = self._get_anchor_point()[:2]
        local_block = self._block_local_anchor()
        joint = pymunk.DampedSpring(
            self.anchor.body,
            self.block.body,
            tuple(anchor_xy),
            (local_block.x, local_block.y),
            self.rest_length,
            self.stiffness,
            self.damping,
        )
        self._spacescene.space.space.add(joint)
        self.spring_joint = joint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_oscillation(self, displacement: float = 0.0, velocity: float = 0.0) -> None:
        """Activate the oscillator with an initial displacement/velocity."""

        if not self._spacescene:
            raise RuntimeError("attach_to_scene must be called before starting motion")

        self._create_joint()
        rest_center = self._get_anchor_point() + RIGHT * (
            self.rest_length + self.block_width / 2
        )
        new_center = rest_center + RIGHT * displacement

        self.block.body.position = (new_center[0], new_center[1])
        self.block.body.velocity = (velocity, 0)
        self.block.body.angular_velocity = 0
        self.block.body.angle = 0
        self.block.body.activate()

    def stop_oscillation(self) -> None:
        """Detach the block from the spring and let it rest."""

        if not self._spacescene:
            return
        if self.spring_joint is not None:
            self._spacescene.space.space.remove(self.spring_joint)
            self.spring_joint = None
        if hasattr(self.block, "body"):
            self.block.body.sleep()

    def reset_displacement(self) -> None:
        """Return the oscillator to the rest position and zero velocity."""

        if not hasattr(self.block, "body"):
            return
        rest_center = self._get_anchor_point() + RIGHT * (
            self.rest_length + self.block_width / 2
        )
        self.block.body.position = (rest_center[0], rest_center[1])
        self.block.body.velocity = (0, 0)
        self.block.body.angular_velocity = 0
        self.block.body.angle = 0
