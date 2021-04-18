using Microsoft.ML.Data;

namespace MLNet
{
    public class SentimentData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool Sentiment { get; set; }

        [LoadColumn(1)]
        public string SentimentText { get; set; }
    }
}
