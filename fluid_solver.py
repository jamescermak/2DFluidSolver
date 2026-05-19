import numpy as np
import bpy #type: ignore

'''
Launches using Blender Dev Extension on VSCODE 'CTRL + SHIFT + P'
'''

class FluidSolver:
    def __init__(self, N=64, diff=0.0001, visc=0.0001, dt=0.1):
        self.N = N
        self.diff = diff
        self.visc = visc
        self.dt = dt
                
        size = (N+2) ** 2
        
        self.u_prev = np.zeros(size) 
        self.v_prev = np.zeros(size) 
        self.dens_prev = np.zeros(size) 
        self.u = np.zeros(size)
        self.v = np.zeros(size)
        self.dens = np.zeros(size)
    
    def inject_at_point(self, point, du, dv, dens=0):
        self.u_prev[point] += du
        self.v_prev[point] += dv
        self.dens_prev[point] += dens

    def clear_sources(self):
        self.u_prev[:] = 0
        self.v_prev[:] = 0
        self.dens_prev[:] = 0
    
    @staticmethod
    def add_source(x, s, dt):
        x += dt*s

    def IX(self, i, j):
        return i + (self.N+2) * j

    def set_bnds(self, b, x):
        N = self.N
        IX = self.IX
        
        for i in range(1, N+1):
            x[IX(0, i)] = -x[IX(1, i)] if b == 1 else x[IX(1, i)]
            x[IX(N+1, i)] = -x[IX(N, i)] if b == 1 else x[IX(N, i)]
            x[IX(i, 0)] = -x[IX(i, 1)] if b == 2 else x[IX(i, 1)]
            x[IX(i, N+1)] = -x[IX(i, N)] if b == 2 else x[IX(i, N)]
        
        x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)])
        x[IX(0, N+1)] = 0.5 * (x[IX(1, N+1)] + x[IX(0, N)])
        x[IX(N+1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N+1, 1)])
        x[IX(N+1, N+1)] = 0.5 * (x[IX(N, N+1)] + x[IX(N+1, N)])

    def diffuse(self, b, x, x0):
        N = self.N
        IX = self.IX
        diff = self.diff
        dt = self.dt
        
        a = dt * diff * N * N
        for k in range(20):
            for i in range(1, N+1):
                for j in range(1, N+1):
                    x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                    x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a)
        self.set_bnds(b, x)
    
    def clamp_to_grid(self, x, y):
        N = self.N
        
        if x < 0.5:
            x = 0.5
        elif x > N + 0.5:
            x = N + 0.5
            
        if y < 0.5:
            y = 0.5
        elif y > N + 0.5:
            y = N + 0.5
            
        return x, y

    def advect(self, b, d, d0, u, v):
        N = self.N
        dt = self.dt
        IX = self.IX
        
        dt0 = dt*N
        for i in range(1, N+1):
            for j in range(1, N+1):
                
                x = i - dt0 * u[IX(i, j)]
                y = j - dt0 * v[IX(i, j)]    
                x, y = self.clamp_to_grid(x, y)
                
                i0 = int(x)
                i1 = i0 + 1
                j0 = int(y)
                j1 = j0 + 1
                
                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1
        
                d[IX(i, j)] = (s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]))
                           
        self.set_bnds(b, d)

    def project(self, u, v, p, div):
        N = self.N
        IX = self.IX
        
        h = 1 / N
        for i in range(1, (N+1)):
            for j in range(1, (N+1)):
                div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i-1, j)] +
                                            v[IX(i, j + 1)] - v[IX(i, j-1)])
                
                p[IX(i, j)] = 0
        
        self.set_bnds(0, div)
        self.set_bnds(0, p)
        
        for k in range(20):
            for i in range(1, (N+1)):
                for j in range(1, (N+1)):
                    p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4
            self.set_bnds(0, p)
        
        for i in range(1, (N+1)):
            for j in range(1, (N+1)):
                u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h
                v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h
                
        self.set_bnds(1, u)
        self.set_bnds(2, v)

    def step(self):
        self.vel_step()
        self.dens_step()
        return self.dens

    def dens_step(self):
        x = self.dens
        x0 = self.dens_prev
        u = self.u
        v = self.v
        
        self.add_source(x, x0, self.dt)
        self.diffuse(0, x=x0, x0=x)
        self.advect(0, x, x0, u, v)

    def vel_step(self):
        u = self.u
        v = self.v
        u0 = self.u_prev
        v0 = self.v_prev
        
        self.add_source(u, u0, self.dt)
        self.add_source(v, v0, self.dt)
        self.diffuse(1, x=u0, x0=u)
        self.diffuse(2, x=v0, x0=v)
        self.project(u, v, u0, v0)
        self.advect(1, u, u0, u0, v0)
        self.advect(2, v, v0, u0, v0)
        self.project(u, v, u0, v0)


class BlenderScene:
    def __init__(self, N):
        self.clear_scene()
        self.reset_handlers()
        self.N = N
        self.fluid_plane = self.make_fluid_plane(N)
        self.fluid_controller = self.make_fluid_controller() 
    
    def clear_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)
    
    def reset_handlers(self):
        for h in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(h)
      
    def make_fluid_plane(self, N):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=N+1, y_subdivisions=N+1)
        bpy.context.object.name = 'Fluid Plane'
        
        mat = bpy.data.materials.get('Fluid') or bpy.data.materials.new('Fluid')
        bpy.context.object.active_material = mat    
        
        BlenderScene._add_dens_attr(bpy.context.object)
        return bpy.context.object
    
    def make_fluid_controller(self):
        bpy.ops.object.empty_add(type='SPHERE', radius=0.1)
        bpy.context.object.name = 'Control'
        return bpy.context.object
 
    def grid_vertices(self):
        vertices = self.fluid_plane.data.attributes['position'].data
        return [point.vector.copy() for point in vertices]
    
    def world_to_gridpoint(self, obj):
        grid_verts = self.grid_vertices()
        find_point = BlenderScene.get_distance(obj.location)
        closest = min(range(len(grid_verts)), key=lambda i: find_point(grid_verts[i]))
        return closest
    
    @staticmethod
    def get_distance(u):
        def calculate(v):
            return np.sqrt((u.x - v.x)**2 + (u.y-v.y)**2)
        return calculate
    
    @staticmethod
    def _add_dens_attr(obj):
        mesh = obj.data
        if 'density' not in mesh.attributes:
            mesh.attributes.new('density', type='FLOAT', domain='POINT')  



class FluidHandler:
    def __init__(self, fs, bs):
        self.f_solver = fs
        self.blender_scene = bs
        self.update_controller_coord()
    
    def __call__(self, *args):
        self.get_from_ui()
        density = self.f_solver.step()
        self.draw_dens(density)
    
    def update_controller_coord(self):
        self.controller_loc = self.blender_scene.fluid_controller.location.copy() 
    
    def get_from_ui(self):
        self.f_solver.clear_sources()
        controller = self.blender_scene.fluid_controller
        gridpoint = self.blender_scene.world_to_gridpoint(controller)
        
        dudt = (controller.location.x - self.controller_loc.x) / self.f_solver.dt
        dvdt = (controller.location.y - self.controller_loc.y) / self.f_solver.dt
        density = 100
        
        self.f_solver.inject_at_point(gridpoint, dudt, dvdt, density)     
        self.update_controller_coord()
    
    def draw_dens(self, density):
        fluid_density = self.blender_scene.fluid_plane.data.attributes['density'].data
        fluid_density.foreach_set('value', density.astype(np.float32))

try:
    N = int(input('Enter N: '))
    fs = FluidSolver(N, diff=0.001, visc=0.1)
    bs = BlenderScene(N)
    f_handler = FluidHandler(fs, bs)
    bpy.app.handlers.frame_change_pre.append(f_handler)

except Exception as e:
    print(e)
