from math import cos, sin, tan, atan, pi, radians
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import matplotlib.image as mpimg


def _gradient_image(w: int, h: int, direction: Optional[int] = 3, noise: float = 0) -> np.ndarray:
    image = np.zeros((w, h), dtype=np.float64)
    # 0: left -> right
    # 1: bottom -> top
    # 2: right -> left
    # 3: top -> bottom
    if direction is None:
        pass
    elif direction == 0:
        for c in range(h):
            image[:, c] = c/h
    elif direction == 1:
        for r in range(w):
            image[r, :] = r/w
    elif direction == 2:
        for c in range(h):
            image[:, h-c-1] = c/h
    elif direction == 3:
        for r in range(w):
            image[w-r-1, :] = r/w

    if noise > 0:
        inoise = noise*np.random.uniform(noise, size=(w, h))
        image += inoise
        image[image > 1] = 1
    return image
# end


class LiquidDrop:
    def __init__(
            self,
            base: float,
            eccentricity: float,
            contact_angle: float | tuple[float, float],
            light_angle: float,
            shadow_bands: int = 4,
            shadow_color: float = .3
    ):
        """

        :param base: length of the contact solid/liquid
        :param eccentricity: ratio between short and long axes
        :param contact_angle: (common | left, right) contact angle in the solid/liquid point
        :param light_angle: angle of the light source
        :param shadow_bands: n of bands used to create the shadow
        :param shadow_color: color of the table
        """
        self.base = base
        self.eccentricity = eccentricity
        self.contact_angle: tuple[float, float] = (
            contact_angle if isinstance(contact_angle, tuple) else (contact_angle, contact_angle)
        )
        self.light_angle = light_angle
        self.shadow_bands = shadow_bands
        self.shadow_color = shadow_color

        left_angle, right_angle = self.contact_angle

        if left_angle < right_angle:
            self.right_liquid_angle = atan(-self.eccentricity/tan(pi-right_angle))
            self.right_major_axis = self.base/2 / cos(self.right_liquid_angle)
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)

            top = self.eccentricity*self.right_major_axis*(1-sin(self.right_liquid_angle))

            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = top/(self.eccentricity*(1-sin(self.left_liquid_angle)))
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)
        elif left_angle > right_angle:
            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = self.base/2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)

            top = self.eccentricity*self.left_major_axis*(1-sin(self.left_liquid_angle))

            self.right_liquid_angle = atan(-self.eccentricity/tan(pi-right_angle))
            self.right_major_axis = top/(self.eccentricity*(1-sin(self.right_liquid_angle)))
            self.right_y = self.eccentricity * self.right_major_axis * sin(self.right_liquid_angle)
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)
            pass
        else:
            self.left_liquid_angle = atan(-self.eccentricity/tan(pi-left_angle))
            self.left_major_axis = self.base/2 / cos(self.left_liquid_angle)
            self.left_y = self.eccentricity * self.left_major_axis * sin(self.left_liquid_angle)
            self.left_x = -self.left_major_axis *cos(self.left_liquid_angle)

            self.right_liquid_angle = self.left_liquid_angle
            self.right_major_axis = self.left_major_axis
            self.right_y = self.left_y
            self.right_x = self.right_major_axis *cos(self.right_liquid_angle)
            pass
        return
    # end

    def x_bounds(self):
        return self.left_x, self.right_x

    def points(self) -> np.ndarray:
        """

        :return:
        """

        left_angles: np.ndarray = np.linspace(pi-self.left_liquid_angle, radians(90))
        right_angles: np.ndarray = np.linspace(radians(90), self.right_liquid_angle)

        left_x = self.left_major_axis*np.cos(left_angles)
        left_y = self.eccentricity*self.left_major_axis*np.sin(left_angles)-self.left_y

        right_x = self.right_major_axis*np.cos(right_angles)
        right_y = self.eccentricity*self.right_major_axis*np.sin(right_angles)-self.right_y

        x = np.concatenate((left_x, right_x))
        y = np.concatenate((left_y, right_y))
        return np.array([x, y])

    def fill(self, ax: Axes, length: float, c='black'):
        self.fill_drop(ax, c=c)
        self.fill_shadow(ax, length=length, c=c)

    def fill_drop(self, ax: Axes, length: float, c='black'):
        xy = self.points()
        ax.fill(xy[0], xy[1], c=c)

    def fill_shadow(self, ax: Axes, length: float, c='black'):
        light_angle = self.light_angle
        xy = self.points()
        if light_angle == 0:
            xy[1] = -length
            xy[1, 0] = 0
            xy[1,-1] = 0
            ax.fill(xy[0], xy[1], c=c)
            return
        # end

        shadow_bands = self.shadow_bands
        shadow_length = 1./tan(radians(light_angle))
        table_color = self.shadow_color

        for i in range(shadow_bands, 0, -1):
            band_length = shadow_length*i/shadow_bands
            fill_color = table_color*(i-1)/shadow_bands
            by = xy.copy()
            by[1] = by[1]*(-band_length)
            by[1].clip(-length, 0)
            ax.fill(by[0], by[1], c=(fill_color, fill_color, fill_color))
# end


class TableShow:
    def __init__(self,
                 table_color: float = 0.3,
                 table_length : float | tuple[float, float] = .1,
                 blur_length: float = .05,
                 xlim: tuple[float,float] = (-1, 1),
                 ylim: tuple[float,float] = (-1, 1)
                 ):
        if not isinstance(table_length, tuple):
            table_length = (0, table_length)
        back_length, front_length = table_length

        self.back_length = back_length
        self.front_length = front_length
        self.blur_length = blur_length
        self.table_color = table_color
        self.xlim = xlim
        self.ylim = ylim

    def fron_length(self):
        return self.front_length

    def fill(self, ax:Axes):
        self.fill_table(ax)
        self.fill_shadow(ax)

    def fill_table(self, ax:Axes):
        xl, xr = self.xlim
        yb, yt = self.ylim
        back_length = self.back_length
        front_length = self.front_length

        img_table = _gradient_image(100, 100, noise=0)*self.table_color
        ax.imshow(img_table, cmap='gray', norm=Normalize(0, 1), extent=(xl, xr, -front_length, back_length))

        if yb < -front_length:
            ax.fill([xl, xl, xr, xr], [-front_length, yb, yb, -front_length], c='black')

    def fill_shadow(self, ax:Axes):
        xl, xr = self.xlim
        back_length = self.back_length
        shadow_length = self.blur_length

        img_shadow = _gradient_image(100, 100, noise=0)* (1-self.table_color) + self.table_color
        ax.imshow(img_shadow, cmap='gray', norm=Normalize(0, 1), extent=(xl, xr, back_length, back_length+shadow_length))
# end


class NoiseShow:
    def __init__(self,
                 noise_color: float = 0.1,
                 noise_band: float = 0.,
                 table_color: float = 0.1,
                 xlim: tuple[float,float] = (-1, 1)
                 ):
        self.xlim = xlim
        self.table_color = table_color
        self.noise_color = noise_color
        self.noise_band = noise_band


    def fill(self, ax:Axes):
        xl, xr = self.xlim
        noise_band = self.noise_band/2
        table_color = self.table_color
        noise_color = self.noise_color/2

        img_noise = mpimg.imread('noise_band.png')

        # normalize
        cmin, cmax = img_noise.min(), img_noise.max()
        img_noise = (img_noise-cmin)/(cmax-cmin)

        # make colors compatible with table color
        nmin = max(table_color-noise_color, 0)
        nmax = min(table_color+noise_color, 1)
        img_noise = img_noise*(nmax-nmin) + nmin

        if noise_band > 0:
            ax.imshow(img_noise, cmap='gray', norm=Normalize(0, 1), extent=(xl, xr, -noise_band, noise_band))
# end


class AxisShow:
    def __init__(self,
                 xlim: tuple[float, float],
                 ylim: tuple[float, float]
                 ):
        self.xlim = xlim
        self.ylim = ylim
        pass

    def fill(self, ax: Axes, axis_off=True):
        ax.set_aspect('equal')
        if axis_off:
            ax.set_axis_off()

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.tight_layout()
# end


def plot_drop(
        *,
        # LiquidDrop
        base: float = 1,
        eccentricity: float = 1.,
        contact_angle: float| tuple[float,float] = 90.,
        light_angle: float = 45.,
        shadow_bands: int = 4,
        shadow_color: float = 0.,
        # TableSHow
        table_color: float = .1,
        table_length: float | tuple[float, float] = .5,
        blur_length: float = 0.,
        # NoiseShow
        noise_color: float = 0,
        noise_band: float = 0,
        # AxisShow
        xlim: (float, float) = (-1, 1),
        ylim: (float, float) = (-1, 1),
        axis_off=True
):
    """

    :param base: length of the drop base
    :param eccentricity: ratio between short and long axes
    :param contact_angle: contact angle in the solid/liquid point
    :param light_angle: angle of the source light
    :param shadow_bands: n of bands used to draw the drop shadow
    :param shadow_color: lighter color of the shadow
    :param table_length: fonta & back length of the table
    :param table_color: lighter color of the table
    :param blur_length: height of the table border shade
    :param noise_color: color of the noisy band in the center of the image
    :param noise_band: height of the noisy band in the center of the image
    :param margins: (left, right, bottom, top)
    """
    ax = plt.gca()

    drop = LiquidDrop(
        base=base,
        eccentricity=eccentricity,
        contact_angle=contact_angle,
        light_angle=light_angle,
        shadow_bands=shadow_bands,
        shadow_color=shadow_color
    )

    table = TableShow(
        xlim=xlim,
        ylim=ylim,
        table_length=table_length,
        table_color=table_color,
        blur_length=blur_length,
    )

    noise = NoiseShow(
        xlim=xlim,
        table_color=table_color,
        noise_color=noise_color,
        noise_band=noise_band
    )

    hax = AxisShow(
        xlim=xlim,
        ylim=ylim
    )

    table.fill(ax)

    noise.fill(ax)

    # drop.fill(ax, table.front_length)
    drop.fill_drop(ax, table.front_length)
    drop.fill_shadow(ax, table.front_length)

    hax.fill(ax, axis_off=axis_off)
# end

# xy=np.zeros(shape=(2, 50))
# xy[0] = np.linspace(-1,1)
# xy[1] = -.5
# xy[1, 0] = 0
# xy[1,49] = 0
#
# plt.fill(xy[0], xy[1], c='orange')
# plt.ylim(-1, 1)
# plt.show()

