from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

URLS = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'pvt_v2_b2': 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth',
    'hardnet68': 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth',
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Download public official/pretrained backbone checkpoints used by the benchmark adapters.')
    parser.add_argument('--output-dir', type=Path, default=Path('weights/official_backbones'))
    parser.add_argument('--models', nargs='*', default=list(URLS.keys()))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        if name not in URLS:
            raise SystemExit(f'Unknown model key: {name}')
        suffix = Path(URLS[name]).name
        dest = args.output_dir / suffix
        print(f'[DOWNLOAD] {name} -> {dest}')
        urllib.request.urlretrieve(URLS[name], dest)
    print('[DONE]')


if __name__ == '__main__':
    main()
