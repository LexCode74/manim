from manimlib import *
import numpy as np


class GradientDescentExplanation(Scene):
    CONFIG = {
        "learning_rate": 0.6,
        "start_x": -3,
        "x_range": (-4, 5),
        "y_range": (0, 10),
    }

    def construct(self):
        axes = self.get_axes()
        graph = axes.get_graph(self.function, color=BLUE)
        graph_label = axes.get_graph_label(graph, Tex("f(x)", color=BLUE))

        title = Text("Descenso del gradiente", font_size=60).to_edge(UP)
        subtitle = Tex(r"x_{n+1} = x_n - \alpha f'(x_n)").next_to(title, DOWN)

        self.play(Write(title))
        self.play(Write(subtitle))
        self.play(ShowCreation(axes), run_time=2)
        self.play(ShowCreation(graph), FadeIn(graph_label))
        self.wait()

        x_tracker = ValueTracker(self.start_x)
        dot = always_redraw(lambda: Dot(color=YELLOW).move_to(
            axes.c2p(x_tracker.get_value(), self.function(x_tracker.get_value()))
        ))
        vertical_line = always_redraw(lambda: axes.get_v_line(
            axes.c2p(x_tracker.get_value(), self.function(x_tracker.get_value())),
            color=YELLOW
        ))
        tangent_line = always_redraw(lambda: self.get_tangent_line(
            axes, x_tracker.get_value(), color=ORANGE
        ))

        current_value_label = always_redraw(lambda: self.get_value_label(
            x_tracker.get_value()
        ))

        self.play(FadeIn(dot), FadeIn(vertical_line))
        self.play(ShowCreation(tangent_line), FadeIn(current_value_label))
        self.wait()

        iterations = 5
        for n in range(iterations):
            new_x = self.next_step(x_tracker.get_value())
            step_text = Tex(
                rf"x_{{{n+1}}} = x_{{{n}}} - {self.learning_rate} \cdot f'(x_{{{n}}})",
                font_size=36,
            )
            step_text.to_corner(DR)
            step_text.set_backstroke(width=4)

            derivative_value = self.derivative(x_tracker.get_value())
            derivative_text = Tex(
                rf"f'(x_{{{n}}}) = {derivative_value:.2f}",
                font_size=36,
            )
            derivative_text.next_to(step_text, UP)
            derivative_text.set_backstroke(width=4)

            self.play(FadeIn(derivative_text))
            self.play(FadeIn(step_text))
            self.play(x_tracker.animate.set_value(new_x), run_time=2)
            self.wait()
            self.play(FadeOut(step_text), FadeOut(derivative_text))

        self.play(FadeOut(tangent_line), FadeOut(vertical_line), FadeOut(current_value_label))
        minimum_dot = Dot(color=GREEN).move_to(
            axes.c2p(1.5, self.function(1.5))
        )
        minimum_label = Tex(r"Mínimo", color=GREEN).next_to(minimum_dot, UR)
        self.play(FadeIn(minimum_dot), Write(minimum_label))
        self.wait(2)

    def function(self, x):
        return (x - 1.5) ** 2 + 1

    def derivative(self, x):
        return 2 * (x - 1.5)

    def get_axes(self):
        axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            width=10,
            height=6,
            axis_config={"include_tip": True},
        )
        axes.to_edge(DOWN)
        axes.add_coordinate_labels()
        return axes

    def get_tangent_line(self, axes, x_value, length=6, color=ORANGE):
        slope = self.derivative(x_value)
        point = axes.c2p(x_value, self.function(x_value))
        direction = np.array([1, slope, 0])
        direction /= np.linalg.norm(direction)
        line = Line(point - length * direction, point + length * direction, color=color)
        return line

    def get_value_label(self, x_value):
        y_value = self.function(x_value)
        label = Tex(
            rf"x = {x_value:.2f}\\\nf(x) = {y_value:.2f}",
            font_size=32,
        )
        label.to_corner(UL)
        label.set_backstroke(width=4)
        return label

    def next_step(self, x_value):
        return x_value - self.learning_rate * self.derivative(x_value)
