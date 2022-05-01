from tkinter.tix import IMAGE
import imageio
import numpy as np
from PIL import Image
from scipy.special import erf
from fluid import Fluid
import cv2
import matplotlib.pyplot as plt
import imageio
import os
import time
from skimage.transform import resize

def source_example(grid_resolution, image_resolution, duration):
    RESOLUTION = grid_resolution
    IMAGE_RESOLUTION = image_resolution
    DURATION = duration

    INFLOW_PADDING = 40
    INFLOW_DURATION = DURATION
    INFLOW_RADIUS = 6
    INFLOW_VELOCITY = 2.5
    INFLOW_COUNT = 8

    fluid_mask = np.zeros(RESOLUTION, dtype=np.uint8)
    cv2.putText(fluid_mask, "SCIVIS", 
                (int(RESOLUTION[0]/3.3), int(RESOLUTION[1]/1.9)), 
                cv2.FONT_HERSHEY_TRIPLEX, 
                1*(RESOLUTION[0] / 200), 
                1, 
                int(2*(RESOLUTION[0] / 200)), 
                cv2.LINE_AA)
    fluid_mask[:,0] = 1
    fluid_mask[0,:] = 1
    fluid_mask[:,-1] = 1
    fluid_mask[-1,:] = 1

    #plt.imshow(fluid_mask)
    #plt.show()
    fluid_mask = np.expand_dims(fluid_mask, 0)
    fluid_mask = np.repeat(fluid_mask, 2, axis=0)
    fluid_mask = fluid_mask > 0


    print('Generating fluid solver, this may take some time.')
    fluid = Fluid(RESOLUTION, IMAGE_RESOLUTION, 
                'dye',
                fluid_mask = fluid_mask)

    center = np.floor_divide(RESOLUTION, 2)
    r = np.min(center) - INFLOW_PADDING

    points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
    points = tuple(np.array((-np.cos(p), np.sin(p))) for p in points)
    normals = tuple(-p for p in points)
    points = tuple(r * p + center for p in points)

    inflow_velocity = np.zeros_like(fluid.velocity)
    inflow_dye = np.zeros(RESOLUTION)
    for p, n in zip(points, normals):
        mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= INFLOW_RADIUS
        inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY
        inflow_dye[mask] = 1
        
    inflow_dye = resize(inflow_dye, 
                        IMAGE_RESOLUTION)
    
    frames = []
    for f in range(DURATION):
        if f <= INFLOW_DURATION:
            fluid.velocity += inflow_velocity
            fluid.dye += inflow_dye
        
        t0 = time.time()
        curl = fluid.step()[1]
        t1 = time.time()
        elapsed_time = t1 - t0
        # Using the error function to make the contrast a bit higher. 
        # Any other sigmoid function e.g. smoothstep would work.
        curl = (erf(curl * 2) + 1) / 4
        curl = resize(curl, IMAGE_RESOLUTION)        
        fluid.dye[fluid_mask[0]] = 0
        color = np.dstack((curl, np.ones(curl.shape), fluid.dye))
        color = (np.clip(color, 0, 1) * 255).astype('uint8')
        im = Image.fromarray(color, mode='HSV').convert('RGB')
        im = np.array(im)
        frames.append(im)
        print(f'Computing frame {f + 1} of {DURATION}, took {elapsed_time : 0.04f} seconds.')

    print('Saving simulation result.')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(dir_path, "..", "Outputs", "sources_example.mp4")
    #frames[0].save(save_path, save_all=True, 
    #               append_images=frames[1:], duration=20, loop=0)
    imageio.mimsave(save_path, frames, fps=30)

def image_example(image_name, grid_resolution, image_resolution, duration):
    RESOLUTION = grid_resolution
    IMAGE_RESOLUTION = image_resolution
    DURATION = duration

    INFLOW_PADDING = 20
    INFLOW_DURATION = DURATION
    INFLOW_RADIUS = 8
    INFLOW_VELOCITY = 1
    INFLOW_COUNT = 3

    dir_path = os.path.dirname(os.path.realpath(__file__))
    load_path = os.path.join(dir_path, "..", "Inputs", image_name)
    img = imageio.imread(load_path)
    img = resize(img, IMAGE_RESOLUTION)

    fluid_mask = np.zeros(RESOLUTION)
    fluid_mask[:,0] = 1
    fluid_mask[0,:] = 1
    fluid_mask[:,-1] = 1
    fluid_mask[-1,:] = 1
    fluid_mask = np.expand_dims(fluid_mask, 0)
    fluid_mask = np.repeat(fluid_mask, 2, axis=0)
    fluid_mask = fluid_mask > 0


    print('Generating fluid solver, this may take some time.')
    fluid = Fluid(RESOLUTION, IMAGE_RESOLUTION, 
                  'dye_r', 'dye_g', 'dye_b',
                fluid_mask = fluid_mask)
    fluid.dye_r = img[:,:,0]
    fluid.dye_g = img[:,:,1]
    fluid.dye_b = img[:,:,2]
    
    points = []
    normals = []
    for i in range(INFLOW_COUNT):
        x = (i+1)/(INFLOW_COUNT+1)
        x *= RESOLUTION[0]
        y = RESOLUTION[1] - INFLOW_PADDING
        points.append(np.array([y,x]))
        normals.append(np.array([-1, 0]))
    points = tuple(points)
    normals = tuple(normals)

    inflow_velocity = np.zeros_like(fluid.velocity)
    
    #inflow_dye = np.zeros(fluid.shape)
    for p, n in zip(points, normals):
        mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= INFLOW_RADIUS
        inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY
        #inflow_dye[mask] = 1

    frames = []
    for f in range(DURATION):
        if f <= INFLOW_DURATION:
            fluid.velocity += inflow_velocity
            #fluid.dye += inflow_dye

        t0 = time.time()
        fluid.step()
        t1 = time.time()
        elapsed_time = t1 - t0
        
        color = np.stack([fluid.dye_r, fluid.dye_g, fluid.dye_b], axis=2)
        #color -= color.min()
        #color /= color.max()
        color *= 255
        color = np.clip(color, 0, 255).astype(np.uint8)
        im = Image.fromarray(color, mode='RGB')
        im = np.array(im)
        frames.append(im)
        print(f'Computing frame {f + 1} of {DURATION}, took {elapsed_time : 0.04f} seconds.')

    print('Saving simulation result.')
    save_path = os.path.join(dir_path, "..", "Outputs", "image_example.mp4")
    #frames[0].save(save_path, save_all=True,
    #               append_images=frames[1:], 
    #               duration=20, loop=0)
    imageio.mimsave(save_path, frames, fps=30)

def mixing_example(grid_resolution, image_resolution, duration):
    RESOLUTION = grid_resolution
    IMAGE_RESOLUTION = image_resolution
    DURATION = duration

    INFLOW_PADDING = 40
    INFLOW_DURATION = DURATION
    INFLOW_RADIUS = 8
    INFLOW_VELOCITY = 4.0

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    fluid_mask = np.zeros(RESOLUTION)
    fluid_mask[:,0] = 1
    fluid_mask[0,:] = 1
    fluid_mask[:,-1] = 1
    fluid_mask[-1,:] = 1
    fluid_mask = np.expand_dims(fluid_mask, 0)
    fluid_mask = np.repeat(fluid_mask, 2, axis=0)
    fluid_mask = fluid_mask > 0


    print('Generating fluid solver, this may take some time.')
    fluid = Fluid(RESOLUTION, IMAGE_RESOLUTION, 
                  'dye_r', 'dye_g', 'dye_b',
                fluid_mask = fluid_mask,
                mode="reflect")
    img = np.zeros([IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 3])
    img[0:int(IMAGE_RESOLUTION[0]/2)] = [227, 150, 190]
    img[int(IMAGE_RESOLUTION[0]/2):] = [174, 224, 244]
    #img = np.random.random(size=[IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 3])
    #img *= 255
    fluid.dye_r = img[:,:,0]
    fluid.dye_g = img[:,:,1]
    fluid.dye_b = img[:,:,2]
    
    frames = []
    
    center = np.floor_divide(RESOLUTION, 2)
    r = np.min(center) - INFLOW_PADDING
    
    for f in range(DURATION):
        if f <= INFLOW_DURATION:
            p = np.array([-np.cos(f/15), np.sin(f/15)])
            p = r * p + center
            
            norm = np.array([np.sin(f/15), np.cos(f/15)])
            mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= INFLOW_RADIUS
            inflow_velocity = np.zeros_like(fluid.velocity)
            inflow_velocity[:,mask] = norm[:,None]
            inflow_velocity *= INFLOW_VELOCITY
            fluid.velocity += inflow_velocity

        t0 = time.time()
        fluid.step()
        t1 = time.time()
        elapsed_time = t1 - t0
        
        color = np.stack([fluid.dye_r, fluid.dye_g, fluid.dye_b], axis=2)
        
        color = np.clip(color, 0, 255).astype(np.uint8)
        im = Image.fromarray(color, mode='RGB')
        im = np.array(im)
        im[mask, :] = [255, 255, 255]
        frames.append(im)
        print(f'Computing frame {f + 1} of {DURATION}, took {elapsed_time : 0.04f} seconds.')

    print('Saving simulation result.')
    save_path = os.path.join(dir_path, "..", "Outputs", "mixing_example.gif")
    #frames[0].save(save_path, save_all=True,
    #               append_images=frames[1:], 
    #               duration=20, loop=0)
    imageio.mimsave(save_path, frames, fps=30)

if(__name__ == "__main__"):
    #source_example([512,512], [512, 512], 500)
    #image_example("GRAVITY.jpg", [512,512], [512, 512], 1500)
    mixing_example([256,256], [256, 256], 1500)
    quit()