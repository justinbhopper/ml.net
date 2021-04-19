using System;
using Microsoft.ML.Transforms;

namespace MLNet.Noshow
{
    [CustomMappingFactoryAttribute("Appointment")]
    public class AppointmentFactory : CustomMappingFactory<AppointmentInput, Appointment>
    {
        public override Action<AppointmentInput, Appointment> GetMapping()
        {
            return (src, dest) => dest.Map(src);
        }
    }
}
