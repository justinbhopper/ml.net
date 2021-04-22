using Microsoft.ML.Data;

namespace MLNet.NoshowV2
{
    public class Appointment
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool NoShow { get; set; }

        [LoadColumn(1)]
        public float LeadTime { get; set; }

        [LoadColumn(2)]
        public float Minutes { get; set; }

        [LoadColumn(3)]
        public bool IsRecurring { get; set; }

        [LoadColumn(4)]
        public bool IsFirstInRecurrence { get; set; }

        [LoadColumn(5)]
        public float Age { get; set; }

        [LoadColumn(6)]
        public bool Male { get; set; }

        [LoadColumn(7)]
        public bool OMBWhite { get; set; }

        [LoadColumn(8)]
        public bool OMBAmericanIndian { get; set; }

        [LoadColumn(9)]
        public bool OMBAsian { get; set; }

        [LoadColumn(10)]
        public bool OMBBlack { get; set; }

        [LoadColumn(11)]
        public bool OMBHawaiian { get; set; }

        [LoadColumn(12)]
        public bool HasEmergencyContact { get; set; }

        [LoadColumn(13)]
        public bool LastAppointmentNoShow { get; set; }

        [LoadColumn(14)]
        public float PreviousNoShows { get; set; }

        [LoadColumn(15)]
        public float TotalScheduled { get; set; }

        [LoadColumn(16)]
        public float NoShowRatio { get; set; }

        [LoadColumn(17)]
        public float LastAppointmentScripts { get; set; }
    }
}
