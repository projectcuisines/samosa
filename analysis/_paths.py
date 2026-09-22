"""Where the analysis scripts find the repository, its figure scripts and the
SAMOSA archive, and a throwaway working copy for scripts that execute figure
scripts (which save their PNG/EPS to the working directory)."""
import atexit, os, shutil, tempfile

REPO       = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
HERE       = os.path.join( REPO, 'analysis' )
FIG_ALL    = os.path.join( REPO, 'figures', 'allcases' )
FIG_SELECT = os.path.join( REPO, 'figures', 'selectcases' )
ARCHIVE    = '/models/data/samosa'

def scratch_copy( src=FIG_ALL ):
    """Copy a figures directory (scripts only) to a temporary directory that is
    removed at exit, and return its path."""
    top = tempfile.mkdtemp( prefix='samosa_analysis_' )
    atexit.register( shutil.rmtree, top, ignore_errors=True )
    dst = os.path.join( top, os.path.basename( src ) )
    shutil.copytree( src, dst, ignore=shutil.ignore_patterns( '*.png', '*.eps', '*.pdf', '__pycache__' ) )
    return dst
