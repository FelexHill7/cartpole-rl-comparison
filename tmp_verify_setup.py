import os
from pathlib import Path
try:
    user_id = os.getuid()
except AttributeError:
    user_id = os.getenv('USERNAME') or 'user'
runtime_dir = Path('/tmp') / f'aicosmos-runtime-{user_id}' if hasattr(os, 'getuid') else Path.cwd() / f'aicosmos-runtime-{user_id}'
runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
runtime_dir.chmod(0o700)
os.environ.setdefault('XDG_RUNTIME_DIR', str(runtime_dir))
print(runtime_dir)
