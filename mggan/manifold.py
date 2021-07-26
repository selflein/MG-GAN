import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


class Manifold:
    def __init__(self, construct_set, radius):
        """

        Args:
            construct_set: Trajectories for construction with shape (num_samples, pred_len, 2)
            radius: Radius for manifold sphere
        """
        self.data = construct_set
        pred_len = construct_set.shape[1]
        self.radius = np.linspace(radius / pred_len, radius, pred_len, endpoint=True)

    def plot_manifold(self, time, color="r", axes=None, border_only=False):
        if axes is None:
            figure, axes = plt.subplots()

        if border_only:
            cmap = plt.get_cmap("Reds", len(time) + 2)
            for i, t in enumerate(time):
                polys = self.get_polygons(t)
                for poly in polys:
                    p = patches.Polygon(
                        np.array(poly.exterior),
                        facecolor="none",
                        edgecolor=cmap(i),
                        lw=3,
                    )
                    p2 = patches.Polygon(
                        np.array(poly.exterior),
                        facecolor=cmap(i),
                        edgecolor="none",
                        lw=3,
                        alpha=0.5,
                        zorder=1,
                    )
                    axes.add_patch(p)
                    axes.add_patch(p2)

        else:
            for idx in range(self.data.shape[0]):
                endpoint = self.data[idx, -1]
                draw_circle = plt.Circle(
                    (endpoint[0], endpoint[1]),
                    self.radius[-1],
                    color=color,
                    fill=False,
                )

                axes.add_artist(draw_circle)
                plt.scatter(endpoint[0], endpoint[1])
        return axes

    def compute_metric(self, test_data):
        """
        Args:
            test_data: Trajectories for construction with shape (num_samples, pred_len,  2)

        Returns:
        """
        is_inside = self.compute_inside(test_data)
        return np.sum(is_inside) / len(test_data)

    def compute_inside(self, test_data):
        is_inside = []
        for idx in range(test_data.shape[0]):
            # Shape (manifold samples, pred_len)
            d = np.linalg.norm(self.data - test_data[idx][None], ord=2, axis=-1)
            condition = d < self.radius[None]
            is_inside.append(condition.any(0).all(0))
        return np.array(is_inside)

    def get_polygons(self, time) -> MultiPolygon:
        if not isinstance(time, list):
            time = [time]
        polys = []
        for time in time:
            for idx in range(self.data.shape[0]):
                endpoint = self.data[idx, time]
                circle = patches.CirclePolygon(
                    (endpoint[0], endpoint[1]), self.radius[time]
                )

                verts = circle.get_path().vertices
                trans = circle.get_patch_transform()
                points = trans.transform(verts)
                polys.append(Polygon(points))
        polys = unary_union(polys)

        if not isinstance(polys, MultiPolygon):
            polys = [polys]
        return polys


if __name__ == "__main__":
    M = Manifold(np.random.rand(12, 20, 2), 0.1)
    print(M.compute_metric(np.random.rand(10, 2)))
    M.plot_manifold([0, 1])
    plt.show()
