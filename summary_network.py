import argparse
# import tensorwatch as tw

from mmcv import Config
# from mmcv.cnn import get_model_complexity_info
# from mmpose.utils.torchstat_utils import model_stats

import sys
sys.path.append('.')
from mmpose.models import build_posenet
from thop import profile
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument('--out-file', type=str,
                        help='Output file name') 
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_posenet(cfg.model)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    
    input = torch.randn(1,3,128,128)
    flops, params = profile(model, inputs=(input, ))
    print('flops:', flops)
    print('params:', params)
    print("params: %.2fMB ------- flops: %.2fMB" % ( params / (1000 ** 2), flops / (1000 ** 2)))  # 这里除以1000的平方，是为了化成M的单位
        
#     df = model_stats(model, input_shape)
#     print(df)
#     if args.out_file:
#         df.to_html(args.out_file + '.html')
#         df.to_csv(args.out_file + '.csv')
#         
#     print('!!!Please be cautious if you use the results in papers. '
#           'You may need to check if all ops are supported and verify that the '
#           'flops computation is correct.')


if __name__ == '__main__':
    main()