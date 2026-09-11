"""JSON interface: python -m matmul_reconstruction input.json > output.json."""
from pathlib import Path
import sys,json
from . import select
from ._core.errors import FallbackError

def main():
    try:
        if len(sys.argv)>2:raise ValueError('Use one input JSON path, or stdin')
        text=Path(sys.argv[1]).read_text(encoding='utf-8')if len(sys.argv)==2 else sys.stdin.read()
        args=json.loads(text)
        allowed={'request','hardware','source_profile','trace','entry','route'}
        if not isinstance(args,dict)or set(args)-allowed:raise ValueError('JSON must contain selector arguments only')
        answer=select(**args)
        json.dump(answer,sys.stdout,ensure_ascii=False,allow_nan=False,indent=2);sys.stdout.write('\n')
    except (ValueError,TypeError,KeyError,OverflowError,FallbackError)as error:
        json.dump({'error_type':type(error).__name__,'error':str(error)},sys.stderr,ensure_ascii=False);sys.stderr.write('\n');return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
