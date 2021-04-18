namespace MLNet.SimulatedNoshow
{
    public class Appointment
    {
        /// <summary>Show/no-show</summary>
        public bool NoShow { get; set; }

        /// <summary>Is this appt on a weekend?</summary>
        public bool Weekend { get; set; }

        /// <summary>It was the Nth season of the year.</summary>
        public float Season { get; set; }

        /// <summary>It was the Nth month of the year.</summary>
        public float Month { get; set; }

        /// <summary>It was the Nth week of the year.</summary>
        public float Week { get; set; }

        /// <summary>It was a major holiday.</summary>
        public bool Holiday { get; set; }

        /// <summary>They had N consecutive no-show appointments just prior.</summary>
        public float PriorNoShows { get; set; }

        /// <summary>They were male/female.</summary>
        public bool Male { get; set; }

        /// <summary>They were N age.</summary>
        public float Age { get; set; }

        /// <summary>They were/werenot married.</summary>
        public bool Married { get; set; }

        /// <summary>They did/didnot live alone.</summary>
        public bool Alone { get; set; }

        /// <summary>Was a medication prescribed in the last session?</summary>
        public bool Medicated { get; set; }

        /// <summary>Is their medication due for a refill?</summary>
        public bool NeedsRefill { get; set; } 
    }
}
