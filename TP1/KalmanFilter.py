import numpy as np


class KalmanFitler:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas) -> None:
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.std_acc = std_acc
        self.x = np.zeros((4, 1))
        self.A = np.identity(4)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.B = np.zeros((4, 2))
        self.B[0, 0] = 0.5 * dt**2
        self.B[1, 1] = 0.5 * dt**2
        self.B[2, 0] = dt
        self.B[3, 1] = dt
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.identity(4)
        self.Q[0, 0] = 0.25 * dt**4 * self.std_acc**2
        self.Q[0, 2] = 0.5 * dt**3 * self.std_acc**2
        self.Q[1, 1] = 0.25 * dt**4 * self.std_acc**2
        self.Q[1, 3] = 0.5 * dt**3 * self.std_acc**2
        self.Q[2, 0] = 0.5 * dt**3 * self.std_acc**2
        self.Q[2, 2] = dt**2 * self.std_acc**2
        self.Q[3, 1] = 0.5 * dt**3 * self.std_acc**2
        self.Q[3, 3] = dt**2 * self.std_acc**2
        self.R = np.identity(2)
        self.R[0, 0] = x_sdt_meas**2
        self.R[1, 1] = y_sdt_meas**2
        self.P = np.identity(self.A.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, Z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (Z - np.dot(self.H, self.x)))
        I = np.identity(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P
        return self.x
