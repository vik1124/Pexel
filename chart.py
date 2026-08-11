from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict
import plotly.graph_objects as go
from pathlib import Path


from config import RESOLUTIONS, CHARTS_DIR


@dataclass
class Chart(ABC):
    chart_type: str
    ground_truth: Dict = field(default_factory=dict)
    fig: go.Figure = None

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

