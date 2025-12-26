# [TODO] this is the reviesd version of blender-exporter-colmap

import numpy as np
from pathlib import Path
import mathutils
from . ext.read_write_model import write_model, Camera, Image
import bpy
from bpy.props import StringProperty, EnumProperty

bl_info = {
    "name": "Scene exporter for colmap",
    "description": "Generates a dataset for colmap by exporting blender camera poses and rendering scene.",
    "author": "Ohayoyogi",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "File/Export",
    "warning": "",
    "wiki_url": "https://github.com/ohayoyogi/blender-exporter-colmap",
    "tracker_url": "https://github.com/ohayoyogi/blender-exporter-colmap/issues",
    "category": "Import-Export"
}

class COLMAP_OT_export_dataset(bpy.types.Operator):
    """Export scene as colmap dataset"""
    bl_idname = "export_scene.colmap_dataset"
    bl_label = "Export Colmap Dataset"

    # Properties
    directory: StringProperty(
        name="Output Directory",
        description="Directory to save the dataset",
        subtype="DIR_PATH"
    )

    export_format: EnumProperty(
        name="Format",
        description="Output format for colmap model",
        items=[
            ('.txt', "Text (.txt)", "Export as text files"),
            ('.bin', "Binary (.bin)", "Export as binary files"),
        ],
        default='.txt'
    )

    camera_model: EnumProperty(
        name="Camera Model",
        description="Camera model to use for export",
        items=[
            ('SIMPLE_PINHOLE', "SIMPLE_PINHOLE", "Simple pinhole model (f, cx, cy)"),
            ('PINHOLE', "PINHOLE", "Pinhole model (fx, fy, cx, cy)"),
            ('OPENCV', "OPENCV", "OpenCV model with distortion (fx, fy, cx, cy, k1, k2, p1, p2)"),
        ],
        default='PINHOLE'
    )

    def invoke(self, context, event):
        # Open the file browser to select the output directory
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        dirpath = Path(self.directory)
        if not dirpath.name:
            dirpath = dirpath.parent

        if not dirpath.exists():
            try:
                dirpath.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                self.report({'ERROR'}, f"Permission denied: cannot create directory {dirpath}")
                return {'CANCELLED'}

        # Start progress bar
        context.window_manager.progress_begin(0, 100)

        try:
            for progress in self.process_dataset(context, dirpath, self.export_format, self.camera_model):
                context.window_manager.progress_update(progress)
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            context.window_manager.progress_end()
            return {'CANCELLED'}
            
        context.window_manager.progress_end()
        self.report({'INFO'}, "Colmap dataset exported successfully")
        return {'FINISHED'}

    def process_dataset(self, context, dirpath: Path, format: str, camera_model: str):
        scene = context.scene
        # Select all cameras in view layer
        scene_cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]

        if len(scene_cameras) == 0:
            raise ValueError("No cameras found in scene")
        
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        images_dir = dirpath / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        cameras = {}
        images = {}

        # Blender (Right, Up, Back) -> OpenCV/COLMAP (Right, Down, Forward)
        coord_trans = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
        
        total_steps = len(scene_cameras) + 1

        for idx, cam in enumerate(sorted(scene_cameras, key=lambda x: x.name)):
            camera_id = idx + 1
            filename = f'{cam.name}.jpg'

            # Intrinsic
            focal_length = cam.data.lens # mm
            sensor_width = cam.data.sensor_width # mm
            sensor_height = cam.data.sensor_height # mm

            # Sensor fit
            if cam.data.sensor_fit == 'VERTICAL':
                s_size = sensor_height
                pixel_size = height
            else:
                s_size = sensor_width
                pixel_size = width

            # Focal length in pixels unit
            f_in_pixels = (focal_length * pixel_size) / s_size

            # Camera parameters by model
            if camera_model == 'SIMPLE_PINHOLE':
                params = [f_in_pixels, width/2, height/2]
            elif camera_model == 'PINHOLE':
                params = [f_in_pixels, f_in_pixels, width/2, height/2]
            elif camera_model == 'OPENCV':
                params = [f_in_pixels, f_in_pixels, width/2, height/2, 0, 0, 0, 0]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")
            
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model,
                width=width, 
                height=height, 
                params=params
            )


            # Extrinsic

            # camera to world matrix
            cam_to_world = cam.matrix_world.copy()
            # flip y and z axes
            cam_to_world_cv = cam_to_world @ coord_trans

            # world to camera matrix for colmap 
            world_to_cam = cam_to_world_cv.inverted()

            # spilit into rotation and translation
            tvec = world_to_cam.translation()
            rot_quat = world_to_cam.to_quaternion()

            images[camera_id] = Image(
                id=camera_id, 
                qvec=np.array([rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z]),
                tvec=np.array([tvec.x, tvec.y, tvec.z]),
                camera_id=camera_id,
                nema=filename, 
                xys=np.array([]),
                point3D_ids=np.array([])
            )


            # Rendering
            context.scene.camera = cam
            
            # Output path
            render_filepath = images_dir / filename
            context.scene.render.filename = str(render_filepath)

            # Render scene
            bpy.ops.render.render(write_still=True)

            yield 100.0 * (idx + 1) / total_steps

        # Write models
        write_model(cameras, images, {}, str(dirpath), format)
        yield 100.0

    
def menu_func_export(self, context):
    self.layout.operator(COLMAP_OT_export_dataset.bl_idname, text="Colmap dataset")

def register():
    bpy.utils.register_class(COLMAP_OT_export_dataset)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(COLMAP_OT_export_dataset)

if __name__ == "__main__":
    register()