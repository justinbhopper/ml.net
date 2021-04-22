﻿using Microsoft.ML.Data;

namespace MLNet.NoshowV2
{
    public class NoShowPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool NoShow { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
