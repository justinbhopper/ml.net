using Microsoft.ML.Data;

namespace MLNet.NoshowV2
{
    public class NoShowPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool NoShow { get; set; }

        /// <summary>
        /// Probability of being no-show (this is not exactly a confidence score of how well its prediction is, because a 1% would mean "very confident its not a no-show")
        /// </summary>
        public float Probability { get; set; }

        /// <summary>
        /// Raw score from the learner (e.g. value before applying sigmoid function to get probability).
        /// </summary>
        public float Score { get; set; }
    }
}
