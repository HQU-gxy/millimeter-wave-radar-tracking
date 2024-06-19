import sys
if sys.platform == "linux":
    from .linux import GPIO
else:
    from .mock import GPIO
