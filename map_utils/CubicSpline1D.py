import numpy as np
import bisect

class CubicSpline1D:
    def __init__(self, x=None, y=None):
        if x is None or y is None:
            # Default constructor
            return

        self.nx = len(x)
        self.a = np.copy(y)
        self.x = np.copy(x)
        self.y = np.copy(y)

        # Compute elementwise difference
        deltas = np.diff(x)

        # Compute matrix a, vector b
        ma = np.zeros((self.nx, self.nx))
        vb = np.zeros(self.nx)
        self._matrix_a(deltas, ma)
        self._vector_b(deltas, vb)

        # Solve for c and copy to attribute vector
        ma_inv = np.linalg.inv(ma)
        tmp_c = np.dot(ma_inv, vb)
        self.c = np.copy(tmp_c)

        # Construct attribute b, d
        self.b = []
        self.d = []
        for i in range(self.nx - 1):
            d_val = (self.c[i + 1] - self.c[i]) / (3.0 * deltas[i])
            b_val = (self.a[i + 1] - self.a[i]) / deltas[i] - deltas[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.d.append(d_val)
            self.b.append(b_val)

    # Calculate the 0th derivative evaluated at t
    def calc_der0(self, t):
        if t < self.x[0] or t >= self.x[-1]:
            return np.nan

        i = self._search_index(t)
        dx = t - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3

    # Calculate the 1st derivative evaluated at t
    def calc_der1(self, t):
        if t < self.x[0] or t >= self.x[-1]:
            return np.nan

        i = self._search_index(t)
        dx = t - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2

    # Calculate the 2nd derivative evaluated at t
    def calc_der2(self, t):
        if t < self.x[0] or t >= self.x[-1]:
            return np.nan

        i = self._search_index(t)
        dx = t - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    # Create the constants matrix a used in spline construction
    def _matrix_a(self, deltas, result):
        result[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != self.nx - 2:
                result[i + 1, i + 1] = 2.0 * (deltas[i] + deltas[i + 1])
            result[i + 1, i] = deltas[i]
            result[i, i + 1] = deltas[i]

        result[0, 1] = 0.0
        result[self.nx - 1, self.nx - 2] = 0.0
        result[self.nx - 1, self.nx - 1] = 1.0

    # Create the 1st derivative vector b used in spline construction
    def _vector_b(self, deltas, result):
        for i in range(self.nx - 2):
            result[i + 1] = 3.0 * ((self.a[i + 2] - self.a[i + 1]) / deltas[i + 1] - (self.a[i + 1] - self.a[i]) / deltas[i])

    # Search the spline for index closest to t
    def _search_index(self, t):
        # return np.searchsorted(self.x, t)
        return bisect.bisect(self.x, t) - 1
