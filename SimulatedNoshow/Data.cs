using System;
using System.Collections.Generic;

namespace MLNet.SimulatedNoshow
{
    public static class Data
    {
        private static readonly Random s_random = new Random();

        /// <param name="count">How many to generate</param>
        /// <param name="outcome">Rules to determine if NoShow</param>
        /// <param name="variance">0 to 1 percentage of whether the outcome rule is ignored.</param>
        public static IEnumerable<Appointment> Train(int count, Func<Appointment, bool> outcome, double variance)
        {
            var results = new List<Appointment>();

            for (var i = 0; i < count; ++i)
            {
                results.Add(Single(item =>
                {
                    item.NoShow = s_random.NextDouble() > variance ? outcome(item) : RandomBool(0.1);
                }));
            }

            return results;
        }

        public static Appointment Single(Action<Appointment> modifier = null)
        {
            var item = new Appointment
            {
                Weekend = RandomBool(2 / 7),
                Male = RandomBool(0.5),
                Alone = RandomBool(0.5),
                Married = RandomBool(0.5),
                Age = s_random.Next(12, 60),
                Holiday = RandomBool(5 / 365),
                Month = s_random.Next(1, 13),
                Week = s_random.Next(1, 53),
                Season = s_random.Next(1, 5),
                NeedsRefill = RandomBool(0.3),
                Medicated = RandomBool(0.3),
                PriorNoShows = RandomBool(0.2) ? s_random.Next(1, 3) : 0,
            };

            modifier?.Invoke(item);

            return item;
        }

        private static bool RandomBool(double chance)
        {
            return s_random.NextDouble() > chance;
        }
    }
}
