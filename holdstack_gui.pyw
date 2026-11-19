"""Dupla kattintással indítható holdstack-felület (konzolablak nélkül)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdstack.gui import main  # noqa: E402

main()
