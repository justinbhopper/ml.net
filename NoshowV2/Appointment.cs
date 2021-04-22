using Microsoft.ML.Data;

namespace MLNet.NoshowV2
{
    public class Appointment
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool NoShow { get; set; }

        [LoadColumn(1)]
        public float Weight { get; set; }

        [LoadColumn(2)]
        public float LeadTime { get; set; }

        [LoadColumn(3)]
        public float DayOfWeek { get; set; }

        [LoadColumn(4)]
        public float Month { get; set; }

        [LoadColumn(5)]
        public float Week { get; set; }

        [LoadColumn(6)]
        public float Hour { get; set; }

        [LoadColumn(7)]
        public float Minutes { get; set; }

        [LoadColumn(8)]
        public bool IsRecurring { get; set; }

        [LoadColumn(9)]
        public bool IsFirstInRecurrence { get; set; }

        [LoadColumn(10)]
        public float Age { get; set; }

        [LoadColumn(11)]
        public bool Male { get; set; }

        [LoadColumn(12)]
        public bool OMBWhite { get; set; }

        [LoadColumn(13)]
        public bool OMBAmericanIndian { get; set; }

        [LoadColumn(14)]
        public bool OMBAsian { get; set; }

        [LoadColumn(15)]
        public bool OMBBlack { get; set; }

        [LoadColumn(16)]
        public bool OMBHawaiian { get; set; }

        [LoadColumn(17)]
        public bool HasEmergencyContact { get; set; }

        [LoadColumn(18)]
        public bool LastAppointmentNoShow { get; set; }

        [LoadColumn(19)]
        public float PreviousNoShows { get; set; }

        [LoadColumn(20)]
        public float TotalScheduled { get; set; }

        [LoadColumn(21)]
        public float NoShowRatio { get; set; }

        [LoadColumn(22)]
        public float LastAppointmentScripts { get; set; }
    }
}
