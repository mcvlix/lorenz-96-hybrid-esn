import numpy as np

class EnKFHybrid:
    def __init__(self, ensemble, obs_func, R, forecast_func):
        """
        Ensemble Kalman Filter using a hybrid forecast model.

        :param ensemble: Initial ensemble (array of shape N_ens x state_dim)
        :param obs_func: Function mapping state to observation space
        :param R: Observation noise covariance matrix
        :param forecast_func: Function to forecast next state
        """
        self.ensemble = ensemble
        self.obs_func = obs_func
        self.R = R
        self.forecast_func = forecast_func

    def predict(self):
        for i in range(len(self.ensemble)):
            self.ensemble[i] = self.forecast_func(self.ensemble[i])

    def update(self, observation):
        Hx = np.array([self.obs_func(e) for e in self.ensemble])
        mean = Hx.mean(axis=0)
        for i in range(len(self.ensemble)):
            self.ensemble[i] += observation - Hx[i] + np.random.multivariate_normal(np.zeros_like(mean), self.R)


class EnKFImperfect:
    def __init__(self, ensemble, obs_func, R, imperfect_rhs, dt=0.01):
        """
        Ensemble Kalman Filter using only the imperfect model (no ESN).

        :param ensemble: Initial ensemble
        :param obs_func: Function mapping state to observation space
        :param R: Observation noise covariance matrix
        :param imperfect_rhs: Function for imperfect model dynamics
        :param dt: Time step
        """
        self.ensemble = ensemble
        self.obs_func = obs_func
        self.R = R
        self.imperfect_rhs = imperfect_rhs
        self.dt = dt

    def predict(self):
        for i in range(len(self.ensemble)):
            dx = self.imperfect_rhs(0, self.ensemble[i])
            self.ensemble[i] += dx * self.dt

    def update(self, observation):
        Hx = np.array([self.obs_func(e) for e in self.ensemble])
        mean = Hx.mean(axis=0)
        for i in range(len(self.ensemble)):
            self.ensemble[i] += observation - Hx[i] + np.random.multivariate_normal(np.zeros_like(mean), self.R)