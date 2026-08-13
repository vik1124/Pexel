from abc import ABC, abstractmethod
from typing import Dict
import plotly.graph_objects as go
from pathlib import Path
from pydantic import BaseModel

from config import RESOLUTIONS, CHARTS_DIR


class Chart(ABC):

    def __init__(self, chart_type: str):
        self.chart_type = chart_type
        self.ground_truth: BaseModel = None
        self.fig: go.Figure = None
        self.extraction_class: type[BaseModel] = None

    @abstractmethod
    def generate_plot(self) -> "Chart":
        """Generate plot, populating self.fig and self.ground_truth."""
        pass

    def save_images(self) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        for res_name, (width, height) in RESOLUTIONS.items():
            subdir = CHARTS_DIR / res_name
            subdir.mkdir(parents=True, exist_ok=True)
            fpath = subdir / f"{self.chart_type}.png"
            self.fig.write_image(str(fpath), width=width, height=height)
            paths[res_name] = fpath
        return paths

