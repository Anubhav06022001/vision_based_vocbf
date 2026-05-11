import numpy as np
import mujoco as mj
import imageio

class VideoRecorder:
    def __init__(self, width , height, fps =30 ):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames= []

    def capture(self, viewport , context ):
         rgb = np.zeros((self.height, self.width, 3), dtype = np.uint8)
         depth = np.zeros((self.height, self.width), dtype = np.float32)

         mj.mjr_readPixels(rgb, depth, viewport, context)
         rgb = np.flipud(rgb)
         
         self.frames.append(rgb)


    def save(self, filename):
        if len(self.frames) == 0:
            print("No frames captured. Nothing to save.")
            return

        imageio.mimsave(
            str(filename),   
            self.frames,
            fps=self.fps,
            codec="libx264"    
        )

        print(f"Saved video: {filename} ({len(self.frames)} frames)")



# class VideoRecorder:
#     def __init__(self, fps=30):
#         self.fps = fps
#         self.frames = []

#     def capture(self, viewport, context):
#         width = viewport.width
#         height = viewport.height

#         rgb = np.zeros((height, width, 3), dtype=np.uint8)
#         depth = np.zeros((height, width), dtype=np.float32)

#         mj.mjr_readPixels(rgb, depth, viewport, context)

#         rgb = np.flipud(rgb)
#         self.frames.append(rgb)

#     def save(self, filename):
#         if len(self.frames) == 0:
#             print("No frames captured.")
#             return

#         imageio.mimsave(
#             str(filename),
#             self.frames,
#             fps=self.fps,
#             codec="libx264"
#         )