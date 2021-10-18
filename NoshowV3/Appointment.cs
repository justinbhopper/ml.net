using Microsoft.ML.Data;

namespace MLNet.NoshowV3
{
    public class Appointment
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool NoShow { get; set; }

        [LoadColumn(1)]
        public float LeadTime { get; set; }

        [LoadColumn(2)]
        public float DayOfWeek { get; set; }

        [LoadColumn(3)]
        public float Hour { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        [LoadColumn(5)]
        public float PreviousNoShows { get; set; }

        [LoadColumn(6)]
        public float TotalScheduled { get; set; }
    }
}
