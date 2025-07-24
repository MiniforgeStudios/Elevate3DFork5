import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class VideoRenderer:
    def __init__(self, vertex_shader_path, normal_fragment_shader_path, obj_mesh, texture_dir, output_dir, video_frame_dir, res=1024):
        self.vertex_shader_path = vertex_shader_path
        self.normal_fragment_shader_path = normal_fragment_shader_path
        self.obj_mesh = obj_mesh
        self.texture_dir = texture_dir
        self.output_dir = Path(output_dir)
        self.video_frame_dir = video_frame_dir
        self.res = res
        self.renderer = self.initialize_renderer()

    def initialize_renderer(self):
        from Projection import HeadlessProjectionMapping
        return HeadlessProjectionMapping(
            vertex_shader_path=self.vertex_shader_path,
            normal_fragment_shader_path=self.normal_fragment_shader_path,
            obj_mesh=self.obj_mesh,
            texture_dir=self.texture_dir,
        )

    def render_video_frames(self, num_frames=360, num_spirals=2, initial_pitch=89.9, final_pitch=-89.9, zoom=1.0):
        yaw_angles = np.linspace(0, 360 * num_spirals, num=num_frames, endpoint=False)
        pitch_angles = np.linspace(initial_pitch, final_pitch, num=num_frames)
        postfixes = []

        for i in range(num_frames):
            yaw = yaw_angles[i] % 360
            pitch = pitch_angles[i]

            postfix = self.render_frame(pitch=pitch, yaw=yaw, zoom=zoom)
            postfixes.append(postfix)

        return postfixes

    def render_frame(self, pitch, yaw, zoom=1.0, thresh=0.6):
        render_kwargs = {
            "pitch": 180 + pitch,
            "yaw": yaw,
            "img_res": (self.res, self.res),
            "zoom": zoom,
        }

        render_funcs = [
            ("render", "rgba"),
            ("render_normal", "normal"),
            ("render_depth", "depth"),
            ("render_cosine", "cosine"),
        ]

        postfix = f"yaw_{yaw:.2f}_pitch_{pitch:.2f}_zoom_{zoom:.2f}"

        for func_name, filename_prefix in render_funcs:
            image = getattr(self.renderer, func_name)(**render_kwargs)
            output_path = self.video_frame_dir / f"{filename_prefix}_{postfix}.png"
            image.save(output_path)

        return postfix

    def create_videos(self, postfixes, fps=30):
        video_paths = {
            "rgba":   self.output_dir / "rgb_video.mp4",
            "normal": self.output_dir / "normal_video.mp4",
            "depth":  self.output_dir / "depth_video.mp4",
        }

        # Determine image dimensions by reading the first image of each type
        first_postfix = postfixes[0]
        image_paths = {key: self.video_frame_dir / f"{key}_{first_postfix}.png" for key in video_paths.keys()}

        # Get dimensions from the first images
        dimensions = {key: cv2.imread(str(image_paths[key])).shape[1::-1] for key in image_paths}
        video_writers = {
            key: cv2.VideoWriter(str(video_paths[key]), cv2.VideoWriter_fourcc(*'mp4v'), fps, dimensions[key])
            for key in video_paths
        }

        # Write each image frame to the video
        for postfix in postfixes:
            for key in video_paths:
                image_path = self.video_frame_dir / f"{key}_{postfix}.png"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(f"Cannot find image {image_path}")
                video_writers[key].write(image)

        # Release all video writers
        for writer in video_writers.values():
            writer.release()

        return video_paths

def main():
    # Configuration and paths
    config = {
        "vertex_shader_path": "./Projection/shaders/vertex_shader.glsl",
        "normal_fragment_shader_path": "./Projection/shaders/normal_fragment_shader.glsl",
        "obj_path":    "/workspace/outputs/MP3D/OmniObject3D/fire_extinguisher_last/20241106_173750/meshes/225.0_45.0.obj",
        "texture_dir": "/workspace/outputs/MP3D/OmniObject3D/fire_extinguisher_last/20241106_173750/textures",
        "output_dir":  "/workspace/outputs/MP3D/OmniObject3D/fire_extinguisher_last/20241106_173750/video",
        "res": 1024,
    }

    renderer = VideoRenderer(**config)

    # Render frames
    postfixes = renderer.render_video_frames(num_frames=360, num_spirals=2, initial_pitch=89.9, final_pitch=-89.9, zoom=1.0 / 1.1)

    # Create videos from frames
    video_paths = renderer.create_videos(postfixes=postfixes, fps=30)

    print("Videos saved:", video_paths)

if __name__ == "__main__":
    main()

