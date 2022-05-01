import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
from skimage.transform import resize
from sklearn.preprocessing import scale
from numerical import difference, operator
import time
import matplotlib.pyplot as plt


class Fluid:
    def __init__(self, shape, output_shape, *quantities, 
                 pressure_order=1, advect_order=3,
                 fluid_mask=None,
                 mode='constant'):
        self.shape = shape

        self.dimensions = len(shape)
        self.fluid_mask = fluid_mask
        self.mode = mode
        
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, np.zeros(output_shape))
            
            
        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))

        laplacian = operator(shape, difference(2, pressure_order))
        self.pressure_solver = factorized(laplacian)
        
        self.advect_order = advect_order

    def step(self):
        advection_map = self.indices - self.velocity

        def advect(field, filter_epsilon=10e-2):
            filtered = spline_filter(field, 
                                     order=self.advect_order, 
                                     mode=self.mode)

            field = filtered * (1 - filter_epsilon) + field * filter_epsilon

            scaled_advection_map = resize(advection_map.transpose([1,2,0]), field.shape)
            scaled_advection_map = scaled_advection_map.transpose([2,0,1])
            scaled_advection_map[0] *= field.shape[0]/advection_map.shape[1]
            scaled_advection_map[1] *= field.shape[1]/advection_map.shape[2]
            f = map_coordinates(field, scaled_advection_map, 
                                   prefilter=False,
                                   order=self.advect_order+2, 
                                   mode=self.mode)
            return f

        
        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])
        
        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))
            
        jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)

        divergence = jacobian.trace()

        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        self.velocity -= np.gradient(pressure)
        self.velocity[self.fluid_mask] = 0
        
        return divergence, curl, pressure
