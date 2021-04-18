using System;
using System.Globalization;

namespace MLNet.Noshow
{
    public class Appointment
    {
        public void Map(AppointmentInput input)
        {
            var date = DateTime.Parse(input.AppointmentDate);
            var time = DateTime.Parse(input.AppointmentTime);
            var created = input.CreateDate == "NULL" ? (DateTime?)null : DateTime.Parse(input.CreateDate);
            var dob = input.DateOfBirth == "NULL" ? (DateTime?)null : DateTime.Parse(input.DateOfBirth);

            Hour = time.Hour;
            CreatedDaysAhead = !created.HasValue ? 0 : ((float)(date - created.Value).TotalDays);
            Weekend = date.DayOfWeek == DayOfWeek.Sunday || date.DayOfWeek == DayOfWeek.Saturday;
            Season = CalcSeason(date);
            Month = date.Month;
            Week = ISOWeek.GetWeekOfYear(date);
            DayOfYear = date.DayOfYear;
            Minutes = input.Minutes;
            Age = !dob.HasValue ? 0 : ((float)(date - dob.Value).TotalDays * 365);
            Sex = input.Sex;
            OMBWhite = input.OMBWhite == "1";
            OMBAmericanIndian = input.OMBAmericanIndian == "1";
            OMBAsian = input.OMBAsian == "1";
            OMBBlack = input.OMBBlack == "1";
            OMBHawaiian = input.OMBHawaiian == "1";
            CDCCode = input.CDCCode;
            HasEmergencyContact = input.HasEmergencyContact == "1";
            LastAppointmentNoShow = input.LastAppointmentShowNoShow != "1";
            PreviousNoShows = input.PreviousNoShows;
            NoShowRatio = input.PreviousNoShows / input.TotalScheduled;
            LastAppointmentScripts = input.LastAppointmentScripts;
            NoShow = input.ShowNoShow == "2";
        }

        private static int CalcSeason(DateTime date)
        {
            var value = date.Month + date.Day / 100f;
            if (value < 3.21 || value >= 12.22) return 3;   // Winter
            if (value < 6.21) return 0; // Spring
            if (value < 9.23) return 1; // Summer
            return 2;   // Autumn
        }

        public float Hour { get; set; }

        public float CreatedDaysAhead { get; set; }

        /// <summary>Is this appt on a weekend?</summary>
        public bool Weekend { get; set; }

        /// <summary>It was the Nth season of the year.</summary>
        public float Season { get; set; }

        /// <summary>It was the Nth month of the year.</summary>
        public float Month { get; set; }

        /// <summary>It was the Nth week of the year.</summary>
        public float Week { get; set; }

        /// <summary>It was the Nth day of the year.</summary>
        public int DayOfYear { get; set; }

        public float Minutes { get; set; }

        /// <summary>They were N age.</summary>
        public float Age { get; set; }

        /// <summary>They were male/female.</summary>
        public string Sex { get; set; }

        public bool OMBWhite { get; set; }

        public bool OMBAmericanIndian { get; set; }

        public bool OMBAsian { get; set; }

        public bool OMBBlack { get; set; }

        public bool OMBHawaiian { get; set; }

        /// <summary>Ethnicity</summary>
        public string CDCCode { get; set; }

        public bool HasEmergencyContact { get; set; }

        public bool LastAppointmentNoShow { get; set; }

        public float PreviousNoShows { get; set; }

        public float NoShowRatio { get; set; }

        public float LastAppointmentScripts { get; set; }

        /// <summary>Show/no-show</summary>
        public bool NoShow { get; set; }
    }
}
