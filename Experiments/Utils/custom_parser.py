import argparse


'''
    Custom parser 
'''
def my_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr", 
        type=float, 
        default=3e-6,
        help="Learning rate for transformers."
        )

    parser.add_argument(
        "--wd", 
        default=0.0, 
        type=float, 
        help="Weight decay for transformers."
        )

    parser.add_argument(
        "--warm-up-steps", 
        type=int, 
        default=0, 
        help="Warm up steps."
        )

    parser.add_argument(
        "--adam-epsilon", 
        default=1e-8, 
        type=float, 
        help="Epsilon for AdamW optimizer."
        )

    parser.add_argument(
        "--bs", 
        type=int, 
        default=8, 
        help="Batch size."
        )

    parser.add_argument(
        "--eval-bs", 
        type=int, 
        default=8, 
        help="Batch size."
        )

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=8, 
        help="Number of epochs."
        )

    parser.add_argument(
        "--name", 
        default="roberta-large", 
        help="Which model."
        )

    parser.add_argument(
        '--shuffle', 
        action='store_true', 
        default=False, 
        help="Shuffle train data such that positive and negative \
        sequences of the same question are not necessarily in the same batch."
        )

    parser.add_argument(
        '--num_choices', 
        type = int,
        default=4, 
        help="Number of possible answers in the dataset."
        )

    parser.add_argument(
        '--max_clarifications', 
        type = int,
        default=3, 
        help="Max number of clarification sentences for the piqa dataset."
        )

    parser.add_argument(
        '--use_categories', 
        type = int,
        default=1, 
        help="Use categories of persian dataset, 1 --> yes, 0 --> no"
        )

    return parser.parse_args()