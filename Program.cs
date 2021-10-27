using System;
using System.Diagnostics;
using System.IO;
using MLNet.NoshowV3;

namespace MLNet
{
    public class Program
    {
        private static readonly string s_rootPath = Path.Combine(Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]), "..", "..", "..");

        public static void Main()
        {
            var stopwatch = Stopwatch.StartNew();
            var program = new Test(s_rootPath);
            //program.Evaluate("model - 46% f1, 63% acc, best recall.zip");
            //program.Evaluate("model - 72% acc.zip");
            //program.Evaluate("model.zip");
            //program.Train();
            program.Predict();

            Console.WriteLine($"Total time took {stopwatch.ElapsedMilliseconds / 1000} seconds.");
        }
    }
}
