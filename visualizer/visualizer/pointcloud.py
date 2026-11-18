from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import os


# default semantic-label -> RGB (0-255) palette used by save_pointcloud_video
# when per-point semantic labels are provided.
DEFAULT_LABEL_COLORS = {
    0: (200, 200, 200),   # background
    1: (255, 128, 0),     # hand
    2: (255, 0, 0),       # hammer / main object
    3: (0, 0, 255),       # nail_board / door_body
    4: (255, 255, 0),     # nail / table
    5: (0, 128, 0),       # table
}

# tab20 fallback palette for label ids not present in DEFAULT_LABEL_COLORS
_TAB20_RGB = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


def visualize_pointcloud(pointcloud, color:tuple=None):
    vis = Visualizer()
    vis.visualize_pointcloud(pointcloud, color=color)
    
class Visualizer:
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # design a colorful point cloud based on 3d coordinates
                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    # Use normalized coordinates as RGB values
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except: # maybe meet NaN error
                    # use simple cyan color
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")

    def save_pointcloud_video(self, pointclouds, video_path, fps=10, width=640, height=480,
                              point_size=3, dpi=100, labels=None, label_colors=None):
        """
        Render a sequence of point clouds into an mp4 video.

        Uses matplotlib (Agg backend, headless) + imageio, so it does NOT require
        kaleido / Chrome / a display. Both of the following inputs are supported:
          - a single np.ndarray of shape (N, 3) or (N, 6)  (treated as one frame)
          - a list of np.ndarray, each (N, 3) or (N, 6), i.e. a frame sequence
            ((N, 6) -> xyz + rgb in [0, 255]; (N, 3) -> xyz, colored by position)

        Args:
            pointclouds: single frame or list of frames.
            video_path: output .mp4 path.
            fps: frames per second.
            width, height: frame size in pixels.
            point_size: matplotlib scatter marker size.
            dpi: rendering dpi.
            labels: optional per-point semantic labels used to colorize the point
                clouds by category (overrides the rgb channels). Can be:
                  - np.ndarray of shape (N,) matching a single-frame pointcloud
                  - a list of np.ndarray, each (N,), one per frame
            label_colors: optional dict mapping label id (int) -> RGB tuple (0-255).
                Defaults to DEFAULT_LABEL_COLORS with a tab20 fallback.
        """
        import io
        import matplotlib
        matplotlib.use('Agg')  # headless, no display / Chrome needed
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d' projection)
        import imageio

        if isinstance(pointclouds, np.ndarray):
            pointclouds = [pointclouds]
        if len(pointclouds) == 0:
            raise ValueError("pointclouds is empty")

        if isinstance(labels, np.ndarray):
            labels = [labels]

        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

        # global bounds so that all frames share the same camera / view
        all_xyz = np.concatenate([pc[:, :3] for pc in pointclouds], axis=0)
        lo = all_xyz.min(axis=0)
        hi = all_xyz.max(axis=0)
        center = (lo + hi) / 2.0
        span = max((hi - lo).max() / 2.0, 1e-6)

        writer = imageio.get_writer(video_path, fps=fps)
        try:
            for i, pc in enumerate(pointclouds):
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                ax = fig.add_subplot(111, projection='3d')
                if labels is not None:
                    # colorize each point by its semantic label, overriding rgb channels
                    pc_labels = labels[i] if isinstance(labels, list) else labels
                    rgb = self._labels_to_colors(pc_labels, label_colors)
                    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=rgb, s=point_size)
                elif pc.shape[1] >= 6:
                    rgb = np.clip(pc[:, 3:6] / 255.0, 0.0, 1.0)
                    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=rgb, s=point_size)
                else:
                    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size)
                ax.set_xlim(center[0] - span, center[0] + span)
                ax.set_ylim(center[1] - span, center[1] + span)
                ax.set_zlim(center[2] - span, center[2] + span)

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi)
                plt.close(fig)
                buf.seek(0)
                frame = imageio.v2.imread(buf)  # (H, W, 3) uint8
                writer.append_data(frame)
        finally:
            writer.close()
        cprint(f"Video saved to {video_path}", 'green')

    def _labels_to_colors(self, labels, label_colors=None):
        """
        Convert per-point semantic labels (int) to an (N, 3) float array in [0, 1]
        using DEFAULT_LABEL_COLORS (or label_colors) with a tab20 fallback.

        Args:
            labels: (N,) int array of semantic labels.
            label_colors: optional dict mapping label id (int) -> RGB tuple (0-255).
        Returns:
            colors: (N, 3) float array in [0, 1].
        """
        labels = np.asarray(labels).reshape(-1)
        colors = np.zeros((labels.shape[0], 3), dtype=np.float64)
        if label_colors is None:
            label_colors = DEFAULT_LABEL_COLORS
        for i, lbl in enumerate(labels):
            key = int(lbl)
            if key in label_colors:
                c = np.asarray(label_colors[key], dtype=np.float64) / 255.0
            else:
                c = np.asarray(_TAB20_RGB[key % len(_TAB20_RGB)], dtype=np.float64) / 255.0
            colors[i] = c
        return colors
    
