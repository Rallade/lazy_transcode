# Core modules for lazy_transcode
import importlib as _il

# Dynamic imports for backward compatibility with unittest.mock.patch
def _im(name):
    return _il.import_module(name)

# Expose modules that were previously at flat paths for legacy patch targets
encoder_config = _im(__name__ + '.config.encoder_config')
file_manager = _im(__name__ + '.processing.file_manager')
