import numpy as np

class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas) -> None:
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.std_acc = std_acc
        self.x = np.zeros((4, 1))
        self.A = np.identity(4)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.B = np.zeros((4, 2))
        self.B[0, 0] = 0.5 * dt ** 2
        self.B[1, 1] = 0.5 * dt ** 2
        self.B[2, 0] = dt
        self.B[3, 1] = dt
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.identity(4)
        self.Q[0, 0] = 0.25 * dt ** 4 * self.std_acc ** 2
        self.Q[0, 2] = 0.5 * dt ** 3 * self.std_acc ** 2
        self.Q[1, 1] = 0.25 * dt ** 4 * self.std_acc ** 2
        self.Q[1, 3] = 0.5 * dt ** 3 * self.std_acc ** 2
        self.Q[2, 0] = 0.5 * dt ** 3 * self.std_acc ** 2
        self.Q[2, 2] = dt ** 2 * self.std_acc ** 2
        self.Q[3, 1] = 0.5 * dt ** 3 * self.std_acc ** 2
        self.Q[3, 3] = dt ** 2 * self.std_acc ** 2
        self.R = np.identity(2)
        self.R[0, 0] = x_sdt_meas ** 2
        self.R[1, 1] = y_sdt_meas ** 2
        self.P = np.identity(self.A.shape[1])

    def predict(self):
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x
    
    def update(self, Z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (Z - self.H @ self.x)
        self.P = self.P - K @ self.H @ self.P
        return self.x