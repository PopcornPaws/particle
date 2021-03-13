Simple particle filter implementation for a 2D robot.

Run `python -m particle.examples.robot2d` from the parent directory of the cloned repo
to see the example animation.

The robot is equipped with a depth sensor that emits rays at given angles and 
provides depth and angle information if the ray reflects from some landmark in the
sensor's range.

The room is almost featureless, except for an open door on the lower wall. The depth sensor
returns a large distance value if it looks through the door as the respective ray
doesn't reflect. Due to the small amount of features, it takes some time for the particles to
converge to the actual position of the robot.

You can observe the variance of the robot's position estimate shrink
(the particles jump together into almost one spot) whenever the robot senses
some relatively unique filter (e.g. corner or the door).
