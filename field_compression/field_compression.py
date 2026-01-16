import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid

class FieldCompression:
    """
    Computes the evolution of a magnetic field under spherically 
    converging or diverging flows. 
    """
    def __init__(self, r0, rf, den0=None, denf=None, mode='trajectory'):
        """
        Handles the computation of the field compression variables 
        (namely, R0/Rf (convergence ratio) and alpha (bending param))

        Parameters
        ----------
        r0 : array_like
            if mode is `trajectory`:
                Starting (Lagrangian) positions of the fluid parcels
                corresponding to the final locations in `rf`
            if mode is `profile`:
                Radial coordinate of associated density profile den0
        
        rf : array_like 
            Final position of the fluid elements. Should correspond to 
            `denf` but not necessarily `r0` (if in profile mode)
        
        den0 : array_like
            Initial density at positions `r0`. Optional if mode is 
            `trajectory`, although improves numerical stability.
        
        denf : array_like
            Final density at positions `rf`. Optional if mode is
            `trajectory`.
        
        mode : str
            Method to compute the field compression variables. Options
            include `trajectory` (compute using starting and final 
            Lagrangian fluid element positions) and `profile` (use the 
            density profiles to compute the Lagrangian evolution)
        """
        
        if mode == 'trajectory':
            self.r0_lagrangian = r0 
            self.den0_lagrangian = den0
            self.rf = rf 
            self.denf = denf 
        elif mode == 'profile':
            # Convert from density profiles to Lagrangian coordinates
            r0_lag, den0_lag = self.convert_to_lagrangian(r0, rf, den0, denf)
            self.r0_lagrangian = r0_lag 
            self.den0_lagrangian = den0_lag
            self.rf = rf 
            self.denf = denf
        else:
            raise ValueError(f'Invalid mode "{mode}". Supported modes include "trajectory" and "profile".')
        
        # Compute the field compression variables
        CR, alpha = self.compute_CR_alpha()
        self.CR = CR 
        self.alpha = alpha
    
    def convert_to_lagrangian(self, r0, rf, den0, denf):
        """Given density profiles, convert to spherical Lagrangian coordinates """

        Menc0 = cumulative_trapezoid(den0 * r0**2, r0, initial=0)
        Mencf = cumulative_trapezoid(denf * rf**2, rf, initial=0)

        r0_lagrangian = np.interp(Mencf, Menc0, r0)
        den0_lagrangian = np.interp(Mencf, Menc0, den0)
        
        # fix r=0 boundary condition to satisfy alpha=1
        r0_lagrangian[0] = (denf[0]*rf[0]**3/den0_lagrangian[0])**(1/3)

        return r0_lagrangian, den0_lagrangian
    
    def compute_CR_alpha(self):
        """Compute the field compression variables alpha and CR. 
        Assumes that the variables are already in Lagrangian coordinates
        """
        r0 = self.r0_lagrangian 
        den0 = self.den0_lagrangian 
        rf = self.rf 
        denf = self.denf

        CR = r0 / rf

        if den0 is not None:
            alpha = 1 - (denf * rf**3) / (den0 * r0**3)
        else:
            # Estimate the Lagrangian Jacobian numerically. Can be unstable
            alpha = 1 - (rf/r0) * np.gradient(r0, rf)

        return CR, alpha

    def evaluate(self, B0, mesh):
        """Evaluate the evolution of starting magnetic field B0 on the
        new `mesh` using the field compression variables CR and alpha.

        Note: there are 3 simultaneous coordinate systems: the spherical
        one corresponding to R0/Rf and alpha, the coordinate system B0
        is on (arb. reg. spaced), and the coordinate system of `mesh`
        (arbitrary structure). 
        To solve this problem, we interpolate both other meshes onto
        the final `mesh`.
        """
        rad, theta, phi = mesh.coords['spherical']

        # Interpolate CR, alpha, and R0 onto the desired mesh
        CR = np.interp(rad, self.rf, self.CR)
        alpha = np.interp(rad, self.rf, self.alpha)
        R0 = np.interp(rad, self.rf, self.r0_lagrangian)

        # Interpolate the B0 field onto the desired mesh, but at R0 (not Rf)
        Br0, Btheta0, Bphi0 = B0.interpolate((R0, theta, phi), "spherical")

        # Equations of field evolution
        Brf = CR**2 * Br0 
        Bthetaf = CR**2 * (1 - alpha) * Btheta0 
        Bphif = CR**2 * (1 - alpha) * Bphi0 

        # Return the B-field on a mesh
        return ArbitraryMesh((rad, theta, phi), (Brf, Bthetaf, Bphif), "spherical")


class ArbitraryMesh:
    """Convenient mesh to store coordinates and fields on. Automatically
    handles coordinate transformations between cartesian, cylindrical,
    and spherical
    """
    def __init__(self, coords, field=None, coordsys='spherical'):
        """Input tuple of coordinates (e.g., (x, y, z)) and designate
        which coordinate system they are associated to
        """
        self.coords = self.coord_transform(coords, coordsys)
        self.base_coordsys = coordsys
        if field is not None:
            self.attach_field(field, coordsys)
        
    def attach_field(self, field, coordsys):
        self.field = self.field_transform(self.coords, field, coordsys)
    
    def coord_transform(self, base_coords, coordsys):
        # https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates
        if coordsys == 'cartesian':
            x, y, z = base_coords
            s = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(s, z)

        elif coordsys == 'cylindrical':
            s, phi, z = base_coords 
            x = s * np.cos(phi)
            y = s * np.sin(phi)
            r = np.sqrt(s**2 + z**2)
            theta = np.arctan2(s, z)
        
        elif coordsys == 'spherical':
            r, theta, phi = base_coords 
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            s = r * np.sin(theta)
        
        else:
            raise ValueError(f'Invalid coordinate system "{coordsys}". Supported modes include "cartesian", "cylindrical" and "spherical".')
        
        coords = {
            'cartesian': (x, y, z),
            'cylindrical': (s, phi, z),
            'spherical': (r, theta, phi)
        }
        return coords
    
    def field_transform(self, coords, base_field, coordsys):
        # https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates
        # Uses the unit vector conversions

        r, theta, phi = coords['spherical']

        if coordsys == 'cartesian':
            Bx, By, Bz = base_field 
            Br = ((Bx * np.sin(theta) * np.cos(phi))
                    + (By * np.sin(theta) * np.sin(phi))
                    + (Bz * np.cos(theta)))
            Bth = ((Bx * np.cos(theta) * np.cos(phi))
                    + (By * np.cos(theta) * np.sin(phi))
                    - (Bz * np.sin(theta)))
            Bphi = (-Bx * np.sin(phi)) + (By * np.cos(phi))
        
        elif coordsys == 'cylindrical':
            Bs, Bphi, Bz = base_field 
            Br = (Bs * np.sin(theta)) + (Bz * np.cos(theta))
            Bth = (Bs * np.cos(theta)) - (Bz * np.sin(theta))
            
        elif coordsys == 'spherical':
            Br, Bth, Bphi = base_field
        
        else:
            raise ValueError(f'Invalid coordinate system "{coordsys}". Supported modes include "cartesian", "cylindrical" and "spherical".')

        Bs = (Br * np.sin(theta)) + (Bth * np.cos(theta))
        Bz = (Br * np.cos(theta)) - (Bth * np.sin(theta))

        Bx = ((Br * np.sin(theta) * np.cos(phi))
                + (Bth * np.cos(theta) * np.cos(phi))
                - (Bphi * np.sin(phi)))
        By = ((Br * np.sin(theta) * np.sin(phi))
                + (Bth * np.cos(theta) * np.sin(phi))
                + (Bphi * np.cos(phi)))
        
        fields = {
            "cartesian": (Bx, By, Bz),
            "cylindrical": (Bs, Bphi, Bz),
            "spherical": (Br, Bth, Bphi)
        }
        return fields

    def interpolate(self, new_coord, new_coordsys):
        """Interpolate a field from its base mesh (and corresponding 
        coordinate system) onto an arbitrarily structured new mesh with
        its own coordinate system
        """
        Binterp = []
        base_coordsys = self.base_coordsys
        
        # Get the new coordinates in the base (regular spaced) coordinate system
        new_coord = self.coord_transform(new_coord, new_coordsys)[base_coordsys]

        x0 = self.coords[base_coordsys][0][:, 0, 0]
        x1 = self.coords[base_coordsys][1][0, :, 0]
        x2 = self.coords[base_coordsys][2][0, 0, :]
        
        for i in range(3):
            interp = RegularGridInterpolator((x0, x1, x2), self.field[new_coordsys][i])
            try:
                Binterp.append(interp(new_coord))
            except ValueError:
                variables = {
                    "cartesian": ('x', 'y', 'z'), 
                    "cylindrical": ('s', 'phi', 'z'), 
                    "spherical": ('r', 'theta', 'phi')
                }
                low0 = new_coord[0] < x0[0]
                low1 = new_coord[1] < x1[0]
                low2 = new_coord[2] < x2[0]
                high0 = new_coord[0] > x0[-1]
                high1 = new_coord[1] > x1[-1]
                high2 = new_coord[2] > x2[-1]
                pt = tuple(np.argwhere(low0 | low1 | low2 | high0 | high1 | high2)[0])
                raise ValueError(
                    f'''At least one of the interpolation points is out of bounds.
                    Bounds on {variables[base_coordsys][0]}: {x0[0]:1.3g}, {x0[-1]:1.3g},
                    Bounds on {variables[base_coordsys][1]}: {x1[0]:1.3g}, {x1[-1]:1.3g},
                    Bounds on {variables[base_coordsys][2]}: {x2[0]:1.3g}, {x2[-1]:1.3g},
                    Point out of bounds: {new_coord[0][pt]:1.3g}, {new_coord[1][pt]:1.3g}, {new_coord[2][pt]:1.3g}
                    '''
                )

        return (Bint for Bint in Binterp)

def capsule_maker(layer_radii, layer_den, output_radii=None):
    """Convenience function for creating a step density profile.

    Parameters
    ----------
    layer_radii : array_like
        Outer radius of each region
    
    layer_den : array_like
        Density to fill each region with
    
    output_radii : array_like
        (Optional) List of radii with which to compute the density at.
        Defaults to None, which creates an evenly spaced array from 
        0 to layer_radii[-1]
    
    Returns
    -------
    output_radii : array_like
        Radii at which the density profile corresponds to
    
    output_den : array_like
        Density profile
    """

    if output_radii is None:
        output_radii = np.linspace(0.1, layer_radii[-1], 400)
    else:
        output_radii = np.array(output_radii)
    
    output_den = layer_den[0] * np.ones_like(output_radii)
    # or, all radii > layer_radii[i] with layer_den[i+1]
    for i in range(len(layer_radii) - 1):
        output_den[output_radii > layer_radii[i]] = layer_den[i+1]
    
    return output_radii, output_den

def create_mesh(extents, npoints, coordsys, indexing='ij'):
    """Creates a 3D coordinate mesh with regular spacing and specified
    extent and resolution

    Parameters
    ----------
    extents : array_like [3 x 2]
        Spatial extent (minimum and maximum) along each of the three 
        dimensions. First index is for the coordinate index, second is 
        for [min, max]
    
    npoints : array_like [3]
        Number of points in each dimension. Should be set to 1 in 
        directions that the mesh does not extend to (e.g., 2D meshes)
    
    coordsys : str
        Coordinate system. Options: `cartesian`, `cylindrical`, 
        `spherical`
    
    indexing : str
        Fortran or C-style indexing. Not actually sure that this works..
    """

    x1 = np.linspace(extents[0][0], extents[0][1], npoints[0])
    x2 = np.linspace(extents[1][0], extents[1][1], npoints[1])
    x3 = np.linspace(extents[2][0], extents[2][1], npoints[2])

    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing=indexing)

    return ArbitraryMesh((X1, X2, X3), coordsys=coordsys)
    