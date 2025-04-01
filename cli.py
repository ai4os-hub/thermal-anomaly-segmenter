"""
Run repository functionalities, namely inference and training,
directly via the command line (CLI) by means of:

"python3 -m cli <command> <arguments>"
"""
import argparse
from marshmallow import missing

from api.schemas import PredArgsSchema, TrainArgsSchema

HL = "======================================"


def main():
    """Coordinate function calls"""

    args = parse_args()
    args_dict = vars(args)

    if args.command == "train":
        from thermal_anomaly_segmenter import train

        print(f"{HL}\nRunning CLI 'training' with\n{args_dict}\n{HL}")
        result = train(**args_dict)

    elif args.command == "predict":
        from thermal_anomaly_segmenter import predict

        print(f"{HL}\nRunning CLI 'prediction' with\n{args_dict}\n{HL}")
        result = predict(**args_dict)

    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(f"{HL}\nResult: {result}\n{HL}")


def parse_args():
    """Parse arguments for CLI call"""

    parser = argparse.ArgumentParser(
        description="CLI for the TASeg (Thermal Anomaly Segmenter)"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # parser for "train" command
    train = subparsers.add_parser(
        "train", help="Train TASeg model."
    )
    train = add_arguments(train, TrainArgsSchema)

    # parser for "predict" command
    predict = subparsers.add_parser(
        "predict", help="Predict with TASeg model."
    )
    predict = add_arguments(predict, PredArgsSchema)

    return parser.parse_args()


def add_arguments(parser, schema):
    """Add arguments to provided parser based on schema"""

    for name, field in schema().fields.items():
        default = field.load_default

        if default == missing and name == "input_file":
            # Replace "input_file" as browsing field
            default = "/srv/thermal-anomaly-segmenter/tests/"\
                      "data/train/KA1_DJI_0_0095_R.npy.lz4"
            parser.add_argument(
                f"--input_filepath", type=str, required=False,
                default=default,
                help=f"{field.metadata['description']} "
                     f"--- DEFAULT: {default}"
            )

        else:
            if hasattr(field.validate, "choices"):
                parser.add_argument(
                    f"--{name}",
                    type=type(default),
                    required=field.required,
                    choices=field.validate.choices,
                    default=default,
                    help=f"{field.metadata['description']} "
                        f"--- DEFAULT: {default}"
                )

            else:
                parser.add_argument(
                    f"--{name}",
                    type=type(default),
                    required=field.required,
                    default=default,
                    help=f"{field.metadata['description']} "
                        f"--- DEFAULT: {default}"
                )

    return parser


if __name__ == "__main__":
    main()
