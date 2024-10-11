import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, L, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, L, N+1)
        self.xij, self.yij = np.meshgrid(x, x, indexing='ij', sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def w(self, c, mx, my):
        """Return the dispersion coefficient"""
        return c*sp.pi*sp.sqrt(mx**2 + my**2)

    def ue(self, c, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w(c, mx, my)*t)

    def l2_error(self, u, ue, dx):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        ue : Sympy function
            The exact solution
        dx : number
            The grid spacing
        """
        return np.sqrt(dx*dx*np.sum(
            (u - sp.lambdify((x, y), ue)(self.xij, self.yij))**2))

    def apply_bcs(self, U):
        U[0] = 0
        U[-1] = 0
        U[:, -1] = 0
        U[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.Nt = Nt;
        self.N = N
        ue = self.ue(c, mx, my)
        u0 = sp.lambdify((x, y), ue.subs({t: 0}))
        #u0 = lambda x, y: np.exp(-40*((x-0.6)**2+(y-0.5)**2))
        L = 1

        self.create_mesh(N, L)
        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        Unm1[:] = u0(self.xij, self.yij)

        dx = L / N
        D = self.D2(N)/dx**2
        dt = cfl*dx/c
        Un[:] = Unm1 + 0.5*(c*dt)**2*(D @ Unm1 + Unm1 @ D.T)
        self.apply_bcs(Un)

        errs = [self.l2_error(Un, ue.subs({t: dt}), dx)]

        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()
        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c*dt)**2*(D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)

            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1

            if store_data == -1:
                errs.append(self.l2_error(Un, ue.subs({t: (n+1)*dt}), dx))
            elif n % store_data == 0:
                plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un

        if store_data == -1:
            return dx, errs
        return self.xij, self.yij, plotdata

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, c, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w(c, mx, my)*t)

    def apply_bcs(self, U):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    if 1:
        sol = Wave2D()
        dx, err = sol(40, 500, cfl=1/np.sqrt(2))
        #print(dx, err[-1], max(err))
        assert(max(err) < 1e-12)
    if 1:
        sol = Wave2D_Neumann()
        dx, err = sol(40, 500, cfl=1/np.sqrt(2))
        #print(dx, err[-1], max(err))
        assert(max(err) < 1e-12)


if __name__ == '__main__':
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    exit()
    sol = Wave2D_Neumann()
    Nt = 100
    xij, yij, data = sol(40, Nt, cfl=1/np.sqrt(2), mx=2, my=2, store_data=2)

    import matplotlib.animation as animation

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('neumannwave.gif', writer='pillow', fps=10)
