import numpy as np

class RobotInput:
    def __init__(self, body_radius: float, wheel_radius: float):
        # Robot parameters
        self.__robot_body_radius = body_radius
        self.__robot_wheel_radius = wheel_radius
        self.right_wheel = 0.0
        self.left_wheel = 0.0

    def body_velocity(self) -> float:
        return self.__robot_wheel_radius * (self.right_wheel + self.left_wheel)

    def body_angular_velocity(self) -> float:
        return self.__robot_body_radius * self.__robot_wheel_radius * (self.right_wheel - self.left_wheel)

class RobotState:
    def __init__(self, x: float, y: float, yaw: float):
        self.__x = x
        self.__y = y
        self.__yaw = yaw

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.__y, self.__yaw])

    def state_transition(self, u : RobotInput, dt: float):
        v = u.body_velocity()
        self.__x += np.cos(self.__yaw) * v * dt
        self.__y += np.sin(self.__yaw) * v * dt
        self.__yaw = normalize_yaw(self.__yaw + u.body_angular_velocity() * dt)

    @staticmethod
    def from_vector(v: np.ndarray):
        return RobotState(v[0], v[1], v[2])

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def yaw(self) -> float:
        return self.__yaw

class GaussianDistribution:
    def __init__(self, mean: float, std_dev: float):
        self.__mean = mean
        self.__a = 2.0 * std_dev**2
        self.__b = std_dev * np.sqrt(2.0 * np.pi)

    def evaluate(self, x: float) -> float:
        return np.exp(-(x - self.__mean)**2 / self.__a) / self.__b

class ParticleFilter:

    def __init__(self, 
            xlim: (float, float),
            ylim: (float, float),
            door_width: float,
            n_samples: int,
            rays: np.ndarray,
            sensor_range: float):

        assert door_width < xlim[1] - xlim[0]

        # Generate initial particles
        x_rand = xlim[0] + (xlim[1] - xlim[0]) * np.random.rand(n_samples)

        y_rand = ylim[0] + (ylim[1] - ylim[0]) * np.random.rand(n_samples)

        psi_rand = -np.pi + 2.0 * np.pi * np.random.rand(n_samples)

        self.__particles = np.vstack((x_rand, y_rand, psi_rand))
        # Generate wall coordinates
        self.__x_min = xlim[0]
        self.__x_max = xlim[1]
        self.__y_min = ylim[0]
        self.__y_max = ylim[1]
        x_mean = (xlim[1] + xlim[0]) / 2.0
        self.__door_left = x_mean - door_width / 2.0
        self.__door_right = x_mean + door_width / 2.0

        self.__wall = np.array([
                [self.__door_right, self.__x_max, self.__x_max, self.__x_min, self.__x_min, self.__door_left],
                [self.__y_min, self.__y_min, self.__y_max, self.__y_max, self.__y_min, self.__y_min]])

        self.__rays = rays
        self.__range = sensor_range
        self.__measurement_likelihood = GaussianDistribution(0.0, 0.7)

    def get_measurement(self, state: RobotState) -> np.ndarray:
        no_return_weight = 5 

        depths = no_return_weight * np.ones_like(self.__rays)
        angles = np.zeros_like(self.__rays)

        if state.x > self.__x_max or state.x < self.__x_min or state.y > self.__y_max or state.y < self.__y_min:
            return np.append(depths, angles)

        for (i, ray) in enumerate(self.__rays):
            r = normalize_yaw(state.yaw + ray)
            tan_r = np.tan(r)
            # check if looking at door
            phi_door_left = np.arctan2(self.__wall[1, -1] - state.y, self.__wall[0, -1] - state.x)
            phi_door_right = np.arctan2(self.__wall[1, 0] - state.y, self.__wall[0, 0] - state.x)
            if r < phi_door_left or r > phi_door_right:
                # not looking at door
                if tan_r > 0.0:
                    if r > 0.0:
                        dx = self.__x_max - state.x
                        dy = self.__y_max - state.y
                    else: 
                        dx = state.x - self.__x_min
                        dy = state.y - self.__y_min
                else: 
                    tan_r = -tan_r
                    if r > 0.0:
                        dx = state.x - self.__x_min
                        dy = self.__y_max - state.y
                    else: 
                        dx = self.__x_max - state.x
                        dy = state.y - self.__y_min

                dx_prime = dy / tan_r
                if dx_prime <= dx:
                    dx = dx_prime
                else:
                    dy = dx * tan_r

                depth = np.sqrt(dx**2 + dy**2)
                if depth < self.__range:
                    depths[i] = depth # if in range, update
            angles[i] = r
        return np.append(depths, angles)

    def resample(self, true_measurements: np.ndarray):
        N = self.__particles.shape[1]
        weights = np.zeros(N)
        cum_sum = 0.0
        for i in range(0, N):
            measurements = self.get_measurement(
                    RobotState(self.__particles[0, i], self.__particles[1, i], self.__particles[2, i]))
            weight = self.__measurement_likelihood.evaluate(np.linalg.norm(true_measurements - measurements))
            cum_sum += weight
            weights[i] = weight

        r = np.random.rand() / N # random value between 0..1/N
        j = 0
        c = weights[0] / cum_sum
        for i in range(0, N):
            n = r + i / N
            while n > c:
                j += 1
                c += weights[j] / cum_sum
            self.__particles[0, i] = self.__particles[0, j]
            self.__particles[1, i] = self.__particles[1, j]
            self.__particles[2, i] = self.__particles[2, j]
                

    def state_update(self, u: RobotInput, dt):
        for i in np.ndindex(self.__particles.shape[1]):
            noise = 2.0 * np.random.rand(3) - 1.0
            new_state = RobotState(self.__particles[0, i], self.__particles[1, i], self.__particles[2, i])
            new_state.state_transition(u, dt)
            self.__particles[0, i] = new_state.x + noise[0] * 0.01
            self.__particles[1, i] = new_state.y + noise[1] * 0.01
            self.__particles[2, i] = normalize_yaw(new_state.yaw + noise[2] * 1e-3)

    @property
    def x_particles(self) -> np.ndarray:
        return self.__particles[0, :]

    @property
    def y_particles(self) -> np.ndarray:
        return self.__particles[1, :]

    @property
    def psi_particles(self) -> np.ndarray:
        return self.__particles[2, :]

    @property
    def x_min(self) -> float:
        return self.__x_min

    @property
    def x_max(self) -> float:
        return self.__x_max

    @property
    def y_min(self) -> float:
        return self.__y_min

    @property
    def y_max(self) -> float:
        return self.__y_max

    @property
    def wall(self) -> np.ndarray:
        return self.__wall

    @property
    def wall_x(self) -> np.ndarray:
        return self.__wall[0, :]

    @property
    def wall_y(self) -> np.ndarray:
        return self.__wall[1, :]

def normalize_yaw(yaw: float) -> float:
    if yaw > np.pi:
        return yaw - 2.0 * np.pi
    elif yaw < -np.pi:
        return yaw + 2.0 * np.pi
    else:
        return yaw
