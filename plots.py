import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).parent

df = pd.read_csv(ROOT_DIR / "cmake-build-debug/results.csv")

plt.gca().set_aspect('equal')
plt.plot(df["x"], df["y"], marker="o")
plt.savefig(ROOT_DIR / "cmake-build-debug/plot.png")