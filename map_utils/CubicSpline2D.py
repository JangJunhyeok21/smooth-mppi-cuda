import numpy as np
from .CubicSpline1D import CubicSpline1D

class CubicSpline2D:
    def __init__(self, global_path=None):
        self.s = []
        self.filtered_points = []

        if global_path is None:
            # Default constructor
            return

        # Filter collinear points from the global path
        self.filtered_points = self.remove_collinear_points(global_path)

        # Calculate the arc length parameter s
        self.calc_s(self.filtered_points[0], self.filtered_points[1])

        # Create cubic splines for x and y
        self.sx = CubicSpline1D(self.s, self.filtered_points[0])
        self.sy = CubicSpline1D(self.s, self.filtered_points[1])

    def setCubicSpline2D(self, global_path):
        self.filtered_points = self.remove_collinear_points(global_path)
        self.calc_s(self.filtered_points[0], self.filtered_points[1])

        self.sx = CubicSpline1D(self.s, self.filtered_points[0])
        self.sy = CubicSpline1D(self.s, self.filtered_points[1])
        print("CubicSpline 1D set")

    # Calculate the s values for interpolation given x, y
    def calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)

        cum_sum = 0.0
        self.s.append(cum_sum)
        for i in range(len(x) - 1):
            cum_sum += np.hypot(dx[i], dy[i])
            self.s.append(cum_sum)

        # Remove any duplicate s values
        self.s = np.unique(self.s).tolist()

    # Calculate the x position along the spline at given t
    def calc_x(self, t):
        return self.sx.calc_der0(t)

    def calc_xdot(self, t):
        return self.sx.calc_der1(t)

    # Calculate the y position along the spline at given t
    def calc_y(self, t):
        return self.sy.calc_der0(t)

    def calc_ydot(self, t):
        return self.sy.calc_der1(t)

    # Calculate the curvature along the spline at given t
    def calc_curvature(self, t):
        dx = self.sx.calc_der1(t)
        ddx = self.sx.calc_der2(t)
        dy = self.sy.calc_der1(t)
        ddy = self.sy.calc_der2(t)

        ref_nom = ddy * dx - ddx * dy
        ref_denom = (dx**2 + dy**2)**1.5

        if abs(ref_nom) < 1e-7:
            ref_nom = 0
        if abs(ref_denom) < 1e-7:
            ref_denom = 1e-7

        k = ref_nom / ref_denom
        return k

    # Calculate the yaw along the spline at given t
    def calc_yaw(self, t):
        dx = self.sx.calc_der1(t)
        dy = self.sy.calc_der1(t)
        yaw = np.arctan2(dy, dx)

        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi

        return yaw

    # Given x, y positions and an initial guess s0, find the closest s value
    def find_s(self, x, y, s0):
        s_closest = s0
        closest = np.inf
        si = self.s[0]

        while si < self.s[-1]:
            if si > self.s[-1]:
                si -= self.s[-1]
            px = self.calc_x(si)
            py = self.calc_y(si)
            dist = np.hypot(x - px, y - py)

            if dist < closest:
                closest = dist
                s_closest = si

            if dist < 0.01:
                return s_closest

            si += 0.01

        return s_closest

    # Given x, y positions and an initial guess s0, find the closest s value within a search range
    def local_find_s(self, x, y, s0, search_range):
        s_closest = s0
        closest = np.inf
        si = s0
        range_val = 0

        while range_val < search_range:
            if si > self.s[-1]:
                si -= self.s[-1]

            px = self.calc_x(si)
            py = self.calc_y(si)
            dist = np.hypot(x - px, y - py)

            if dist < closest:
                closest = dist
                s_closest = si

            if dist < 0.01:
                return s_closest

            range_val += 0.01
            si = s0 + range_val

        return s_closest
    
    # def sd_to_xy(self, s, d):
    #     px = self.calc_x(s)
    #     py = self.calc_y(s)

    #     dx_dtheta = self.calc_xdot(s)
    #     dy_dtheta = self.calc_ydot(s)

    #     width = abs(d)
    #     if (d > 0): # right
    #         x = px + width * dy_dtheta
    #         y = py - width * dx_dtheta
    #     else: # left
    #         x = px - width * dy_dtheta
    #         y = py + width * dx_dtheta

    #     return x, y

    def sd_to_xy(self, s, d):
        px = self.calc_x(s)
        py = self.calc_y(s)

        dx = self.calc_xdot(s)
        dy = self.calc_ydot(s)

        # right-normal = (dy, -dx), signed offset by d
        return px + d * dy, py - d * dx

    # Calculate the lateral error at given x, y, s
    def calc_lateral_deviation(self, x, y, s0):
        px = self.calc_x(s0)
        py = self.calc_y(s0)

        dx_dtheta = self.calc_xdot(s0)
        dy_dtheta = self.calc_ydot(s0)

        lateral_error = (x - px) * dy_dtheta - (y - py) * dx_dtheta
        return lateral_error

    # Remove any collinear points from a list of points by the triangle rule
    def remove_collinear_points(self, global_path):
        x_ = [global_path[0].x, global_path[1].x]
        y_ = [global_path[0].y, global_path[1].y]

        for i in range(2, len(global_path) - 1):
            # collinear = self.are_collinear(global_path[i - 2].x, global_path[i - 2].y,
            #                                global_path[i - 1].x, global_path[i - 1].y,
            #                                global_path[i].x, global_path[i].y)
            # if collinear:
            #     continue
            x_.append(global_path[i].x)
            y_.append(global_path[i].y)

        # Ensure the last point is included
        x_.append(global_path[-1].x)
        y_.append(global_path[-1].y)

        return [x_, y_]

    # Determine if 3 points are collinear using the triangle area rule
    def are_collinear(self, x1, y1, x2, y2, x3, y3):
        area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        return abs(area) <= 1e-10  # smaller value -> bigger filtered points
