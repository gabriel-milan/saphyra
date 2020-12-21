__all__ = []

# Check if root is installed
has_root=False
try:
    import ROOT
    has_root=True
except:
    print('ROOT not installed at your system. correction table class not available.')

from . import create_jobs
__all__.extend( create_jobs.__all__ )
from .create_jobs import *

#from . import crossval_table
#__all__.extend( crossval_table.__all__ )
#from .crossval_table import *

from . import reprocess
__all__.extend( reprocess.__all__ )
from .reprocess import *

from . import model_generator_base
__all__.extend( model_generator_base.__all__ )
from .model_generator_base import *

from . import plot_generator
__all__.extend( plot_generator.__all__ )
from .plot_generator import *


#if has_root:
#from . import correction_table
#__all__.extend( correction_table.__all__ )
#from .correction_table import *



