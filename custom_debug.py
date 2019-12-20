import os.path
import random

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

tf_v1 = tf.compat.v1
tf_v2 = tf.compat.v2


class HparamWriter:
    """
    A utility class for logging hyper-parameters for the TensorBoard HParam
    plugin. If it is used at the beginning of the training routine, also
    failed runs will appear on in the TensorBoard.

    Parameters
    ----------
    model_dir : str
        The model dir.
    hparams : dict
        A dictionary of ``<hyper-parameter-name : hyper-parameter-value>``
        pairs. ``hyper-parameter-name`` is a string, ``hyper-parameter-value``
        can be one of

        * ``int``
        * ``float``
        * ``str``
        * ``bool``
    metrics : list of str
        A list of the summary names of the metrics that shall be shown next
        to the hyper parameters. The metric names must be fully qualified, i.e.
        also include their parent scopes.
    eval_name : str or None
        The name of the evaluation folder. `tf.estimator.Estimator.eval_dir
        <https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#eval_dir>`_
        can help predicting the *name*. Defaults to *eval*.
    use_v2 : bool
        Use the TensorFlow 2 API.

        .. warning::
           Currently not working.
    """

    def __init__(self, model_dir, hparams, metrics, eval_name=None, use_v2=False):
        self._model_dir = model_dir
        self._hparams = hparams
        self._eval_name = eval_name or "eval"
        self._metrics = [hp.Metric(metric, group=self._eval_name) for metric in metrics]
        self._experiment_dir, self._trial_id = os.path.split(self._model_dir)
        self._use_v2 = use_v2
        self._hparams_written = False

    @staticmethod
    def _write_v1_summary(logdir, summary):
        writer = tf_v1.summary.FileWriterCache.get(logdir=logdir)
        writer.add_summary(summary.SerializeToString())
        writer.flush()

    def _write_v1(self):
        # Write the hparams for this trial
        if not self._hparams_written:
            self._write_v1_summary(
                logdir=self._model_dir,
                summary=hp.hparams_pb(self._hparams, trial_id=self._trial_id),
            )
            self._hparams_written = True

        # Check if the global hparams config already exists
        tf.io.gfile.makedirs(self._experiment_dir)
        if tf.io.gfile.glob(os.path.join(self._experiment_dir, "events.*")):
            tf.logging.debug("Experiment-wide hparams config already exists.")
        else:  # if not, create it
            self._write_v1_summary(
                logdir=self._experiment_dir,
                summary=hp.hparams_config_pb(
                    hparams=list(self._hparams.keys()), metrics=self._metrics
                ),
            )

    def _write_v2(self):
        with tf_v2.summary.create_file_writer(
                logdir=self._experiment_dir
        ).as_default():
            hp.hparams_config(hparams=list(self._hparams.keys()), metrics=self._metrics)

        # Write the hparams for this trial
        with tf_v2.summary.create_file_writer(
                logdir=self._model_dir
        ).as_default():
            hp.hparams(self._hparams, trial_id=self._trial_id)

    def write(self):
        """
        Actually writes the HParams.
        """
        if not self._use_v2:
            self._write_v1()
        else:
            raise NotImplementedError("V2 HParam logging not functional yet")


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

    with tf.Graph().as_default() as graph:
        tf.summary.scalar(METRIC_LOSS, tf_v1.random.uniform((), name="t_loss"))
        tf.summary.scalar(METRIC_ACC1, tf_v1.random.uniform((), name="t_accuracy/top_5"))
        tf.summary.scalar(METRIC_ACC5, tf_v1.random.uniform((), name="t_accuracy/top_5"))
        summary_op = tf_v1.summary.merge_all()

    for i in range(NUM_RUNS):
        session_dir = os.path.join(BASE_LOGDIR, "%03d" % i)
        print(f"Session directory: {session_dir}")
        hparam_writer = HparamWriter(
            model_dir=session_dir,
            hparams={h: h.domain.sample_uniform(rng) for h in (HP_LR, HP_OPTIMIZER)},
            metrics=[METRIC_LOSS, METRIC_ACC1, METRIC_ACC5]
        )
        hparam_writer.write()
        for mode in MODES:
            logdir = os.path.abspath(os.path.join(session_dir, mode))
            print(f"Log dir: {logdir}")
            with tf.Session(graph=graph) as session:
                for step in range(NUM_STEPS):
                    summary_writer = tf_v1.summary.FileWriterCache.get(logdir=logdir)
                    summary = session.run(summary_op)
                    summary_writer.add_summary(summary, global_step=step)
                    summary_writer.flush()


if __name__ == '__main__':
    main()
