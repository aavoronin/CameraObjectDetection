import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
from PIL import Image


@dataclass
class CircleFeature:
    """Represents a colored circle feature"""
    x: int  # X position (x0 + random offset)
    y: int  # Y position (y0 + random offset)
    radius: int  # Circle radius (independent random value)
    color: Tuple[int, int, int]  # RGB color tuple


class ImageArea:
    def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.x0: int = x0  # Left boundary
        self.y0: int = y0  # Top boundary
        self.x1: int = x1  # Right boundary
        self.y1: int = y1  # Bottom boundary
        self.image: Optional[Image.Image] = None
        self.features: List[CircleFeature] = []

    def area_width(self):
        return int(self.x1 - self.x0)

    def area_height(self):
        return int(self.y1 - self.y0)

    def save_image(self, image: Image.Image) -> None:
        """Stores an image in this area"""
        self.image = image

    def add_random_circles(self, n_circles: int) -> None:
        """Adds n random circles with independent position/radius"""
        min_radius: int = min(self.x1 - self.x0, self.y1 - self.y0) // 20  # minimum radius
        max_radius: int = min(self.x1 - self.x0, self.y1 - self.y0) // 5  # Maximum radius

        max_x: int = self.x1 - self.x0  # Maximum x offset
        max_y: int = self.y1 - self.y0  # Maximum y offset

        self.features = []  # Clear existing features

        for _ in range(n_circles):
            # Generate positions starting from x0/y0
            x: int = self.x0 + random.randint(0, max_x)
            y: int = self.y0 + random.randint(0, max_y)

            # Generate radius completely independently
            radius: int = random.randint(min_radius, max_radius)

            color: Tuple[int, int, int] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

            self.features.append(CircleFeature(x, y, radius, color))