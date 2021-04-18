using Microsoft.ML.Data;

namespace MLNet
{
    public class NotePrediction
    {
        [ColumnName("PredictedLabel")]
        public string NoteType;
    }
}
