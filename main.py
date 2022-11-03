import argparse
from utils.param_config import ParamConfig
from models.gni_acc import run_gnia
from models.dropout_acc import run_drop
from models.bnn_acc import run_bnn
from models.dgp_acc import run_dgp
from models.fpn_acc import run_fpn, run_fnn_fnn_mlp, run_rdm_fnn_fc
from models.dnn_acc import run_dnn
from models.fnn_acc import run_fnn
from models.gp_acc import run_gp


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--m",
        default="fpn",
        metavar="model",
        help="the name of processing dataset",
    )
    parser.add_argument(
        "--d",
        default="gsad",
        metavar="Dataset",
        help="the name of processing dataset",
    )
    parser.add_argument(
        "--c",
        default="cuda:0",
        metavar="Device",
        help="the device that used for the model running",
    )
    parser.add_argument(
        "--nl",
        type=float,
        default=0.0,
        help="noise level on dataset corruption",
    )
    parser.add_argument(
        "--sig",
        type=float,
        default=0.1,
        help="GNI parameter",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.1,
        help="dropout rate",
    )
    parser.add_argument(
        "--inference",
        default='svi',
        help="inference",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="n_samples",
    )
    parser.add_argument(
        "--n_rule",
        type=int,
        default=5,
        help="n_rule",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="warmup",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.001,
        help="step_size",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=40,
        help="num_steps",
    )
    parser.add_argument(
        "--svi_lr",
        type=float,
        default=0.01,
        help="svi_lr",
    )
    parser.add_argument(
        "--svi_n_epoch",
        type=int,
        default=200,
        help="svi_n_epoch",
    )
    parser.add_argument(
        "--lr_dgp",
        type=float,
        default=0.001,
        help="lr_dgp",
    )
    parser.add_argument(
        "--n_epoch_dgp",
        type=int,
        default=30,
        help="n_epoch_dgp",
    )

    args = parser.parse_args()
    param_config = ParamConfig()
    param_config.config_parse(f"{args.d}_config")
    if args.c != "cuda:0":
        param_config.device = args.c
    param_config.noise_level = args.nl

    if args.m == "gnia":
        param_config.gni_sigma = args.sig
        run_gnia(param_config)
    elif args.m == "dropout":
        param_config.drop_rate = args.dr
        run_drop(param_config)
    elif args.m == "bnn":
        param_config.inference = args.inference
        param_config.n_samples = args.n_samples
        param_config.warmup = args.warmup
        param_config.step_size = args.step_size
        param_config.num_steps = args.num_steps
        param_config.lr_svi = args.svi_lr
        param_config.n_epoch_svi = args.svi_n_epoch
        run_bnn(param_config)
    elif args.m == "dgp":
        param_config.lr_dgp = args.lr_dgp
        param_config.n_epoch_dgp = args.n_epoch_dgp
        run_dgp(param_config)
    elif args.m == "fnn":
        run_fnn(param_config)
    elif args.m == "gp":
        run_gp(param_config)
    elif args.m == "dnn":
        param_config.n_rules = args.n_rule
        run_dnn(param_config)
    elif args.m == "fpn":
        param_config.n_rules = args.n_rule
        run_fpn(param_config)
    elif args.m == "rfnn_mlp":
        run_fnn_fnn_mlp(param_config)
    elif args.m == "rfnn_fc":
        run_rdm_fnn_fc(param_config)
    # run_bnn(param_config)


if __name__ == "__main__":
    main()
