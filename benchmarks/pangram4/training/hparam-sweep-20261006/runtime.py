from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
# Trackio and its pure-Python dependencies go last so they never shadow the training stack.
sys.path.append(str(ROOT/'vendor-trackio'))
SPACE_CACHE=str(ROOT/'assets')
TRACKIO_PROJECT='pangram-hparam-sweep-20261006'
def require_space():
 if str(ROOT)!='/tmp/pangram-hparam-sweep-20261006':
  raise RuntimeError('Model operations restricted to the training Space sweep folder')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('TRACKIO_DIR',str(ROOT/'trackio'))
