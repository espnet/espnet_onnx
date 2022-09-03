from .frontend import Frontend

# If torch is installed, then load TorchFrontend
try:
    from .torch_frontend import TorchFrontend
except ImportError:
    pass
