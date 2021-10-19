using Microsoft.ML.Data;

namespace MLNet
{
    public static class MlNetExtensions
    {
        /// <summary>
        /// Computes F-Beta score.
        /// </summary>
        /// <param name="threshold">When > 1, recall is more favored.  When < 1, precision is more favored.</param>
        /// <seealso>https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc</seealso>
        /// <seealso>https://machinelearningmastery.com/fbeta-measure-for-machine-learning/</seealso>
        public static double FBeta(this BinaryClassificationMetrics metrics, double threshold = 0.5)
        {
            var betaSqrd = threshold * threshold;
            return (1 + betaSqrd) * metrics.PositivePrecision * metrics.PositiveRecall / (betaSqrd * metrics.PositivePrecision + metrics.PositiveRecall);
        }
    }
}
