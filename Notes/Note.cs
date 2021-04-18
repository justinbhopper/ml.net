using Microsoft.ML.Data;

namespace MLNet
{
    public class Note
    {
        [LoadColumn(0)]
        public string Agency { get; set; }
        [LoadColumn(1)]
        public string NoteType { get; set; }
        [LoadColumn(2)]
        public string Location { get; set; }
        [LoadColumn(3)]
        public string Reason { get; set; }
    }
}
