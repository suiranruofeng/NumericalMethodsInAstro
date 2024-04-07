import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

# prep figure
fig = plt.figure(figsize=(7, 9))
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot(grid[0:2, 0])
ax2 = plt.subplot(grid[2, 0])

ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.set_aspect("equal", "box")
ax1.set_xticks([-2, -1, 0, 1, 2])
ax1.set_yticks([-2, -1, 0, 1, 2])
ax2.set_aspect(0.007)
ax2.set_xlabel("time")
ax2.set_ylabel("energy")

(sc1,) = ax1.plot([], [], "o", markersize=1, color=[0.7, 0.7, 1])
(sc2,) = ax1.plot([], [], "o", markersize=10, color="blue")
(sc3,) = ax2.plot([], [], color="red", markersize=1, label="KE")
(sc4,) = ax2.plot([], [], color="blue", markersize=1, label="PE")
(sc5,) = ax2.plot([], [], color="black", markersize=1, label="Etot")

ax2.legend(loc="upper right")


def getAcc(pos, mass, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law
        pos  is an N x 3 matrix of positions
        mass is an N x 1 vector of masses
        G is Newton's Gravitational constant
        softening is the softening length
        a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = dx**2 + dy**2 + dz**2 + softening**2
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def getEnergy(pos, vel, mass, G):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE


"""N-body simulation"""

# Simulation parameters
N = 50  # Number of particles
t = 0  # current time of the simulation
tEnd = 20.0  # time at which simulation ends
dt = 0.02  # timestep
softening = 0.1  # softening length
G = 1.0  # Newton's Gravitational Constant

# Generate Initial Conditions
rng = np.random.default_rng(20240402)  # set the random number generator seed

mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
pos = rng.normal(0, 1, (N, 3))  # randomly selected positions and velocities
vel = rng.normal(0, 1, (N, 3))

# Convert to Center-of-Mass frame
vel -= np.mean(mass * vel, 0) / np.mean(mass)

# calculate initial gravitational accelerations
acc = getAcc(pos, mass, G, softening)

# calculate initial energy of system
KE, PE = getEnergy(pos, vel, mass, G)

# number of timesteps
Nt = int(np.ceil(tEnd / dt))

# save energies, particle orbits for plotting trails
pos_save = np.zeros((N, 3, Nt + 1))
pos_save[:, :, 0] = pos
KE_save = np.zeros(Nt + 1)
KE_save[0] = KE
PE_save = np.zeros(Nt + 1)
PE_save[0] = PE
t_all = np.arange(Nt + 1) * dt
ax2.set(xlim=(0, tEnd), ylim=(-300, 300))


def run(i):
    # Simulation Main Loop
    # (1/2) kick
    global vel, pos, acc, t

    vel += acc * dt / 2.0

    # drift
    pos += vel * dt

    # update accelerations
    acc = getAcc(pos, mass, G, softening)

    # (1/2) kick
    vel += acc * dt / 2.0

    # update time
    t += dt

    # get energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # save energies, positions for plotting trail
    pos_save[:, :, i + 1] = pos
    KE_save[i + 1] = KE
    PE_save[i + 1] = PE

    xx = pos_save[:, 0, max(i - 50, 0) : i + 1]
    yy = pos_save[:, 1, max(i - 50, 0) : i + 1]
    sc1.set_data(xx, yy)
    sc2.set_data(pos[:, 0], pos[:, 1])

    sc3.set_data(t_all[: i + 1], KE_save[: i + 1])
    sc4.set_data(t_all[: i + 1], PE_save[: i + 1])
    sc5.set_data(t_all[: i + 1], KE_save[: i + 1] + PE_save[: i + 1])


i = np.arange(Nt)

plt.show(FuncAnimation(fig, run, i, interval=dt * 1000))

