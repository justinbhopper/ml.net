namespace MLNet.NoShowSim
{
    public class FeatureContribution
    {
        public FeatureContribution(string name, float value)
        {
            Name = name;
            Value = value;
        }

        public string Name { get; }

        public float Value { get; }
    }
}
