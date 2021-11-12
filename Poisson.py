import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

class Poisson(object):

    def __init__(self, shape, dx, rhoi, rhoj, rhok, phi_0, limit):

        self.shape = shape
        self.dx = dx
        self.rhoi = rhoi
        self.rhoj = rhoj
        self.rhok = rhok
        self.phi_0 = phi_0
        self.limit = limit

        self.create_phi_lattice()
        self.create_rho_lattice()

########## Create Initial lattices ##########

    """Create the lattice containing values of phi"""
    def create_phi_lattice(self):

        self.phi_lattice = np.zeros((self.shape, self.shape, self.shape))
        for i in range(self.shape):
            for j in range(self.shape):
                for k in range(self.shape):

                    if  (i!=0) and (i!=self.shape-1) and (j!=0) and (j!=self.shape-1) and (k!=0) and (k!=self.shape-1):
                        self.phi_lattice[i][j] = np.random.randint(-10,11)/100.0 + self.phi_0

    """Create the lattice containing values of rho"""
    def create_rho_lattice(self):

        self.rho_lattice = np.zeros((self.shape, self.shape, self.shape))
        self.rho_lattice[self.rhoi][self.rhoj][self.rhok] = 1.0

########## Update ##########

    def laplacian(self, grid):

        kernal =  [[[0., 0., 0.,],
                        [0., 1., 0.],
                        [0., 0., 0.]],

                        [[0., 1., 0.],
                        [1., 0., 1.],
                        [0., 1., 0.]],

                        [[0., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 0.]]]

        return signal.fftconvolve(grid, kernal, mode='same')

    """Update phi using convoluation"""
    def convolve_phi(self):

        #Store phi from the previous time step for use in convergance calculation
        self.phi_lattice_old = np.copy(self.phi_lattice)
        self.phi_lattice = (1/6.0) * self.laplacian(self.phi_lattice) + (1/6.0)*self.rho_lattice
        self.phi_lattice_new = np.copy(self.phi_lattice)

    def run_animation_in_file(self):

        for s in range(1000):
            self.convolve_phi()
            #self.convergance_check()

    def animate(self, i):
        self.run_animation_in_file()
        self.mat.set_data(self.phi_lattice[:, :, self.rhok])
        print(i)
        return self.mat,

    def run(self):
        fig, ax = plt.subplots()
        self.mat = ax.imshow(self.phi_lattice[:,:,self.rhok], interpolation = 'gaussian', cmap='magma')
        fig.colorbar(self.mat)
        ani = FuncAnimation(fig, self.animate, interval = 1, blit = True)

        plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 9:
        print("Incorrect Number of Arguments Presented.")
        print("Usage: " + sys.argv[0] + "lattice Size, dx, i, j, k, phi_0, limit, Data/Animate")
        quit()
    else:
        shape = int(sys.argv[1])
        dx = float(sys.argv[2])
        i = int(sys.argv[3])
        j = int(sys.argv[4])
        k = int(sys.argv[5])
        phi_0 = float(sys.argv[6])
        limit = float(sys.argv[7])
        type = str(sys.argv[8])


    model = Poisson(shape,  dx, i, j, k, phi_0, limit)
    if type in ['D', 'd', 'data', 'Data']:
        model.convergance_check()
    elif type in ['A', 'a', 'animate', 'Animate']:
        model.run()
