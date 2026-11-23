import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(str(backend_dir))

if __name__ == "__main__":
    import runpy
    runpy.run_path(str(backend_dir / "test_pipeline.py"), run_name="__main__")
