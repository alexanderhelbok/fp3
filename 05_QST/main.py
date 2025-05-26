import numpy as np
import QuantumTomography as qKLib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


def append_to_output(output, names, counts):
    for n, c in zip(names, counts):
        A1, B1, A2, B2 = 0, 0, 0, 0
        n1 = n[0]
        n2 = n[1]
        if n1 == "H":
            A1 = 1
            B1 = 0
        elif n1 == "V":
            A1 = 0
            B1 = 1
        elif n1 == "A":
            A1 = 0.7071
            B1 = 0.7071
        elif n1 == "D":
            A1 = 0.7071
            B1 = -0.7071
        elif n1 == "L":
            A1 = 0.7071
            B1 = 0.7071j
        elif n1 == "R":
            A1 = 0.7071
            B1 = -0.7071j
        if n2 == "H":
            A2 = 1
            B2 = 0
        elif n2 == "V":
            A2 = 0
            B2 = 1
        elif n2 == "A":
            A2 = 0.7071
            B2 = 0.7071
        elif n2 == "D":
            A2 = 0.7071
            B2 = -0.7071
        elif n2 == "L":
            A2 = 0.7071
            B2 = 0.7071j
        elif n2 == "R":
            A2 = 0.7071
            B2 = -0.7071j
        output.append([1, 0, 0, c, A1, B1, A2, B2])
    return output


output = []

def load_data(input, output):
    if input == 1:
        # HH
        names = ["HH", "VH", "HV", "VV",]
        angles_hwp = [[None, None], [40, None], [None, 43], [40, 43]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [134, 12945, 11068, 599]
        append_to_output(output, names, counts)

        #HD
        names = ["HD", "VD", "HA", "VA",]
        angles_hwp = [[None, 20], [40, 20], [None, 65.5], [40, 65.5]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [7265, 3854, 1549, 12073]
        append_to_output(output, names, counts)

        #DH
        names = ["DH", "DV", "AH", "AV",]
        angles_hwp = [[17.5, None], [17.5, 43], [62.5, None], [62.5, 43]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [5189, 8856, 6786, 5098]
        append_to_output(output, names, counts)

        #DD
        names = ["DD", "DA", "AD", "AA",]
        angles_hwp = [[17.5, 20], [17.5, 65.5], [62.5, 20], [62.5, 65.5]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [1661, 10690,  9403, 1936]
        append_to_output(output, names, counts)

        #HR
        names = ["HR", "HL", "VR", "VL",]
        angles_hwp = [[None, None], [None, None], [40, None], [40, None]]
        angels_qwp = [[None, 42.5], [None, 132.5], [None, 42.5], [None, 132.5]]
        counts = [5504, 6509, 5906, 6209]
        append_to_output(output, names, counts)

        #RH
        names = ["RH", "RV", "LH", "LV",]
        angles_hwp = [[None, None], [None, 40], [None, None], [None, 40]]
        angels_qwp = [[42.5, None], [42.5, None], [132.5, None], [132.5, None]]
        counts = [5115, 7027, 5553, 6681]
        append_to_output(output, names, counts)

        #DR
        names = ["DR", "DL", "AR", "AL",]
        angles_hwp = [[17.5, None], [17.5, None], [62.5, None], [62.5, None]]
        angels_qwp = [[None, 42.5], [None, 132.5], [None, 42.5], [None, 132.5]]
        counts = [4208, 9450, 7450, 2397]
        append_to_output(output, names, counts)

        #RD
        names = ["RD", "RA", "LD", "LA",]
        angles_hwp = [[None, 20], [None, 65.5], [None, 20], [None, 65.5]]
        angels_qwp = [[40, None], [40, None], [130, None], [130, None]]
        counts = [9305, 3261, 3679, 7675]
        append_to_output(output, names, counts)

        #RR
        names = ["RR", "RL", "LR", "LL",]
        angles_hwp = [[None, None], [None, None], [None, None], [None, None]]
        angels_qwp = [[40, 42.5], [40, 132.5], [130, 42.5], [130, 132.5]]
        counts = [1446, 9573, 10458, 1745]
        append_to_output(output, names, counts)
    elif input == 2:
        # HH
        names = ["HH", "VH", "HV", "VV",]
        angles_hwp = [[None, None], [40, None], [None, 43], [40, 43]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [161, 8555, 13752, 1105]
        append_to_output(output, names, counts)

        #HD
        names = ["HD", "VD", "HA", "VA",]
        angles_hwp = [[None, 20], [40, 20], [None, 65.5], [40, 65.5]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [8137, 2319, 5542, 6989]
        append_to_output(output, names, counts)

        #DH
        names = ["DH", "DV", "AH", "AV",]
        angles_hwp = [[17.5, None], [17.5, 43], [62.5, None], [62.5, 43]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [4193, 9699, 4863, 3619]
        append_to_output(output, names, counts)

        #DD
        names = ["DD", "DA", "AD", "AA",]
        angles_hwp = [[17.5, 20], [17.5, 65.5], [62.5, 20], [62.5, 65.5]]
        angels_qwp = [[None, None], [None, None], [None, None], [None, None]]
        counts = [1759, 9907, 9019, 1029]
        append_to_output(output, names, counts)

        #HR
        names = ["HR", "HL", "VR", "VL",]
        angles_hwp = [[None, None], [None, None], [40, None], [40, None]]
        angels_qwp = [[None, 42.5], [None, 132.5], [None, 42.5], [None, 132.5]]
        counts = [5814, 7152,  5725, 3655]
        append_to_output(output, names, counts)

        #RH
        names = ["RH", "RV", "LH", "LV",]
        angles_hwp = [[None, None], [None, 40], [None, None], [None, 40]]
        angels_qwp = [[42.5, None], [42.5, None], [130, None], [132.5, None]]
        counts = [4655, 6766, 3609, 7043]
        append_to_output(output, names, counts)

        #DR
        names = ["DR", "DL", "AR", "AL",]
        angles_hwp = [[17.5, None], [17.5, None], [62.5, None], [62.5, None]]
        angels_qwp = [[None, 42.5], [None, 132.5], [None, 42.5], [None, 132.5]]
        counts = [6707, 6019, 4150, 4091]
        append_to_output(output, names, counts)

        #RD
        names = ["RD", "RA", "LD", "LA",]
        angles_hwp = [[None, 20], [None, 65.5], [None, 20], [None, 65.5]]
        angels_qwp = [[40, None], [40, None], [130, None], [130, None]]
        counts = [5617, 5461, 5989, 4201]
        append_to_output(output, names, counts)

        #RR
        names = ["RR", "RL", "LR", "LL",]
        angles_hwp = [[None, None], [None, None], [None, None], [None, None]]
        angels_qwp = [[40, 42.5], [40, 132.5], [130, 42.5], [130, 132.5]]
        counts = [787, 9833, 9579, 878]
        append_to_output(output, names, counts)

load_data(2, output)

tomo_input = np.array(output)
intensity = np.ones(len(output))
t = qKLib.Tomography()
rho, intens, fval = t.state_tomography(tomo_input, intensity)

# calculate fidelity
psip = np.array([0, 1, 1, 0])
psim = np.array([0, 1, -1, 0])
phip = np.array([1, 0, 0, 1])
phim = np.array([1, 0, 0, -1])

for state in [psip, psim, phip, phim]:
    print("Fidelity with state {}: {}".format(state, qKLib.fidelity(rho, state)))

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax = [ax1, ax2]

ax[0].set_title(r"$Re(\rho)$", fontsize=20)
ax[1].set_title(r"$Im(\rho)$", fontsize=20)

for axis in ax:
    # turn off the grid
    axis.grid(False)
    # turn off shading
    axis.xaxis.pane.fill = False
    axis.yaxis.pane.fill = False
    axis.zaxis.pane.fill = False

    # zaxis on left
    axis.zaxis.set_ticks_position('lower')

    # set ticks to bra ket HV
    axis.set_xticks([-0.5, 0.5, 1.5, 2.5])
    axis.set_xticklabels([r"$\vert HH\rangle$", r"$\vert HV\rangle$", r"$\vert VH\rangle$", r"$\vert VV\rangle$"])
    axis.set_yticks([-0.5, 0.5, 1.5, 2.5])
    axis.set_yticklabels([r"$\vert HH\rangle$", r"$\vert HV\rangle$", r"$\vert VH\rangle$", r"$\vert VV\rangle$"])

    # fill square from -0.5, -0.5 to 2.5, 2.5
    min_x, max_x = -0.5, 3
    x = np.array([[min_x, min_x], [max_x, max_x]])
    y = np.array([[min_x, max_x], [min_x, max_x]])
    z = np.array([[0, 0], [0, 0]])

    # semi-transparent plane at z=0
    axis.plot_surface(x, y, z, alpha=0.4, color='black')

    # plot wire at z = 0.5 and -0.5
    axis.plot_wireframe(x, y, np.ones((2, 2))*0.5, color='red', alpha=0.1)
    axis.plot_wireframe(x, y, np.ones((2, 2))*-0.5, color='blue', alpha=0.1)

    axis.set_zlim(-0.7, 0.7)
    axis.set_zticks([-0.5, 0, 0.5])

# create two density matrices with positive and negative eigenvalues for plotting
rho1 = rho.copy()
for i, rho in enumerate([np.real(rho), np.imag(rho)]):
    mask1 = np.where(rho > 0)
    mask2 = np.where(rho < 0)
    rho1, rho2 = rho.copy(), rho.copy()
    rho1[mask1] = None
    rho2[mask2] = None

    n = np.size(rho)
    xpos, ypos = np.meshgrid(range(rho.shape[0]), range(rho.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)

    ax[i].bar3d(xpos, ypos, 0, 0.5, 0.5, rho1.flatten(), lw=2, ls="-", edgecolor='C4', linewidth=1, color="k", alpha=0.3)
    ax[i].bar3d(xpos, ypos, 0, 0.5, 0.5, rho2.flatten(), lw=2, ls="-", edgecolor='C4', linewidth=1, color="k")


plt.tight_layout()
# plt.savefig("density_matrix.pdf", bbox_inches='tight')
plt.show()
