import json
import argparse


def main(inp_args):
    channels_cfg = [8 * (2 ** i) for i in range(inp_args.num_back_layers)]
    block_configs = [2 for _ in range(inp_args.num_back_layers)]

    model_cfg = {
        "backbone_cfg": {
            "out_shape": (inp_args.inp_shape, inp_args.inp_shape),
            "channel_configs": channels_cfg,
            "block_configs": block_configs,
            "inp_ch": inp_args.inp_ch
        },
        "vit_cfg": {
            "img_size": (inp_args.inp_shape, inp_args.inp_shape),
            "patch_size": (2, 2),
            "emb_dim": 64,
            "mlp_dim": 64,
            "num_heads": 8,
            "num_layers": inp_args.n_vit_layers,
            "n_classes": inp_args.n_cls,
            "dropout_rate": inp_args.dropout,
            "at_d_r": 0.0,
            "inp_ch": channels_cfg[-1]
        }
    }

    with open(inp_args.path_to_save, "w") as f:
        json.dump(model_cfg, f)


if __name__ == "__main__":
    desc = "Vision Transformer"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--path_to_save', type=str, default="model_cfg.json", help='path to json file in which params '
                                                                                   'will be saved')

    parser.add_argument('--inp_shape', type=int, default=64, help='shape of output of backbone layer - must be '
                                                                  'divisible by 2')

    parser.add_argument('--num_back_layers', type=int, default=4, help='number of layers in backbone')

    parser.add_argument('--inp_ch', type=int, default=3, help='number of expected channels in input image')

    parser.add_argument('--n_cls', type=int, default=10, help='number of output classes')

    parser.add_argument('--n_vit_layers', type=int, default=5, help='number of layers in vit transformer')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for vit transformer')

    args = parser.parse_args()

    main(args)
