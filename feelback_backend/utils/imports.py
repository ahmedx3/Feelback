# Boilerplate to Relative import some package from parent directory
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[3]))
__old_package__ = __package__
__package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])


from ...feelback import utils as feelback_utils
from ...feelback.utils import video_utils
from ...feelback import Feelback


__package__ = __old_package__
sys.path.remove(str(file.parents[3]))
