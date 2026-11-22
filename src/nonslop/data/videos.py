from collections.abc import Callable
from pathlib import Path
from shutil import copyfile
from threading import RLock


class ArtifactStore:
    def __init__(
        self, root: Path, serving_root: Path, reload: Callable[[], None]
    ) -> None:
        self.root = root
        self.serving_root = serving_root
        self.reload = reload
        self.lock = RLock()
        serving_root.mkdir(parents=True, exist_ok=True)

    def materialize(self, relative_path: Path) -> Path:
        with self.lock:
            destination = self.serving_root / relative_path
            if not destination.is_file():
                self.reload()
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial")
                copyfile(self.root / relative_path, temporary)
                temporary.replace(destination)
            return destination
