try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except:
    __version__ = "0.1.0"
