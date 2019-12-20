import os
import random

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


tf.compat.v1.enable_eager_execution()
tf_v2 = tf.compat.v2


HP_LR = hp.HParam("learning_rate", hp.RealInterval(0.01, 0.1))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "sgd"]))

MODES_WORKING = ("train", "eval")
MODES_ISSUE = ("", "eval")
# TODO(wchargin): try both things here
MODES = MODES_ISSUE

METRIC_LOSS = "loss"
METRIC_ACC1 = "accuracy/top_1"
METRIC_ACC5 = "accuracy/top_5"

NUM_RUNS = 20
NUM_STEPS = 10
BASE_LOGDIR = "logs"


def main():
    rng = random.Random(0)

    for i in range(NUM_RUNS):
        session_dir = os.path.join(BASE_LOGDIR, "%03d" % i)
        print(f"Session directory: {session_dir}")
        with tf_v2.summary.create_file_writer(session_dir).as_default():
            hp.hparams(
                {h: h.domain.sample_uniform(rng) for h in (HP_LR, HP_OPTIMIZER)}
            )
        for mode in MODES:
            if mode:
                logdir = os.path.join(session_dir, mode)
            else:
                logdir = session_dir
            print(f"Log dir: {logdir}")
            with tf_v2.summary.create_file_writer(logdir).as_default():
                for step in range(NUM_STEPS):
                    tf_v2.summary.scalar(METRIC_LOSS, rng.random(), step=step)
                    tf_v2.summary.scalar(METRIC_ACC1, rng.random(), step=step)
                    tf_v2.summary.scalar(METRIC_ACC5, rng.random(), step=step)


if __name__ == "__main__":
    main()
