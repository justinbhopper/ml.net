using Microsoft.ML.Data;

namespace MLNet.Noshow
{
    public class AppointmentInput
    {
        [LoadColumn(0)]
        public string AppointmentDate { get; set; }

        [LoadColumn(1)]
        public string AppointmentTime { get; set; }

        [LoadColumn(2)]
        public float Minutes { get; set; }

        [LoadColumn(3)]
        public string CreateDate { get; set; }

        [LoadColumn(4)]
        public string ShowTime { get; set; }

        [LoadColumn(5)]
        public float ShowNoShow { get; set; }

        [LoadColumn(6)]
        public string IsRecurring { get; set; }

        [LoadColumn(7)]
        public string IsFirstInRecurrence { get; set; }

        [LoadColumn(8)]
        public string ClientKey { get; set; }

        [LoadColumn(9)]
        public string DateOfBirth { get; set; }

        [LoadColumn(10)]
        public string Sex { get; set; }

        [LoadColumn(11)]
        public string SexOrientKey { get; set; }

        [LoadColumn(12)]
        public string OMBWhite { get; set; }

        [LoadColumn(13)]
        public string OMBAmericanIndian { get; set; }

        [LoadColumn(14)]
        public string OMBAsian { get; set; }

        [LoadColumn(15)]
        public string OMBBlack { get; set; }

        [LoadColumn(16)]
        public string OMBHawaiian { get; set; }

        [LoadColumn(17)]
        public string CDCCode { get; set; }

        [LoadColumn(18)]
        public string HasEmergencyContact { get; set; }

        [LoadColumn(19)]
        public string LastAppointmentShowNoShow { get; set; }

        [LoadColumn(20)]
        public float PreviousNoShows { get; set; }

        [LoadColumn(21)]
        public float TotalScheduled { get; set; }

        [LoadColumn(22)]
        public float LastAppointmentScripts { get; set; }
    }
}
