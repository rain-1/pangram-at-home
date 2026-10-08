from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
# Optional extra package dirs placed ahead of vendor, e.g. a PEFT version matching an external checkpoint
# (PANGRAM_EXTRA_PATH=/tmp/final-20261007/vendor-peft021:/tmp/final-20261007/vendor for woog's H200-trained MoE).
for _p in reversed([x for x in os.environ.get('PANGRAM_EXTRA_PATH','').split(':') if x]):
 sys.path.insert(0,_p)
# Trackio and its pure-Python dependencies go last so they never shadow the training stack.
sys.path.append(str(ROOT/'vendor-trackio'))
SPACE_CACHE=str(ROOT/'assets')
TRACKIO_PROJECT='pangram-nemotron-20261007'
def require_space():
 if str(ROOT)!='/tmp/pangram-nemotron-20261007':
  raise RuntimeError('Model operations restricted to the training Space sweep folder')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('TRACKIO_DIR',str(ROOT/'trackio'))
