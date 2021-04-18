using Microsoft.ML.Data;

namespace MLNet.Noshow
{
    public class NoShowPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool NoShow { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public float[] FeatureContributions { get; set; }
    }
}
