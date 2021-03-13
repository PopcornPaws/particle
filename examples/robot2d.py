from ..particle import ParticleFilter, RobotInput, RobotState
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# depth sensor parameters
angle_of_sight = np.radians(15)
rays = np.linspace(-angle_of_sight, angle_of_sight, 5)
sensor_range = 3 

pf = ParticleFilter(
        xlim = (-2.0, 2.0),
        ylim = (-1.5, 1.5),
        door_width = 0.1,
        n_samples = 1000,
        rays = rays,
        sensor_range = sensor_range)

robot = RobotState(0.0, 0.0, 0.9)
measurements = pf.get_measurement(robot)
landmarks_x = [robot.x + d * np.cos(ray + robot.yaw) for (d, ray) in zip(measurements, rays)]
landmarks_y = [robot.y + d * np.sin(ray + robot.yaw) for (d, ray) in zip(measurements, rays)]

fig, ax = plt.subplots()
ax.plot(pf.wall_x, pf.wall_y, linewidth = 3, color = "k")
pscatter = ax.scatter(pf.x_particles, pf.y_particles, s = 1.0, zorder = 3)
rscatter = ax.scatter(robot.x, robot.y, s = 30, color = "g", zorder = 1)
mscatter = ax.scatter(landmarks_x, landmarks_y, s = 10, marker = "x", color = "c", zorder = 3)
left_ray, = ax.plot(\
         [robot.x, robot.x + sensor_range * np.cos(rays[0] + robot.yaw)],
         [robot.y, robot.y + sensor_range * np.sin(rays[0] + robot.yaw)], color = "r", linestyle = "--", linewidth = 0.8)
right_ray, = plt.plot(\
         [robot.x, robot.x + sensor_range * np.cos(rays[-1] + robot.yaw)],
         [robot.y, robot.y + sensor_range * np.sin(rays[-1] + robot.yaw)], color = "r", linestyle = "--", linewidth = 0.8)
ax.set_xlabel("x coordinate [m]")
ax.set_ylabel("y coordinate [m]")
ax.grid(True)

u = RobotInput(1.15, 0.025)
u.right_wheel = 30 
u.left_wheel = 30 
dt = 0.01

def animate(i):
   if i == 0:
      u.left_wheel = 205 
      u.right_wheel = -205
   elif i == 25:
      u.left_wheel = 35
      u.right_wheel = 35
   elif i == 50:
      u.left_wheel = -105
      u.right_wheel = 105
   elif i == 75:
      u.left_wheel = -45
      u.right_wheel = -45

   pf.state_update(u, dt)
   robot.state_transition(u, dt)

   measurements = pf.get_measurement(robot)
   landmarks_x = [robot.x + d * np.cos(ray + robot.yaw) for (d, ray) in zip(measurements[0:len(rays)], rays)]
   landmarks_y = [robot.y + d * np.sin(ray + robot.yaw) for (d, ray) in zip(measurements[0:len(rays)], rays)]

   # resample based on measurement update weights
   pf.resample(measurements)

   rscatter.set_offsets([robot.x, robot.y])
   pscatter.set_offsets(np.c_[pf.x_particles, pf.y_particles])
   mscatter.set_offsets(np.c_[landmarks_x, landmarks_y])

   left_ray.set_data([
         [robot.x, robot.x + sensor_range * np.cos(rays[0] + robot.yaw)],
         [robot.y, robot.y + sensor_range * np.sin(rays[0] + robot.yaw)]])
   right_ray.set_data([
         [robot.x, robot.x + sensor_range * np.cos(rays[-1] + robot.yaw)],
         [robot.y, robot.y + sensor_range * np.sin(rays[-1] + robot.yaw)]])

anim = animation.FuncAnimation(fig, animate, frames = 100, interval = 100)
plt.show()
