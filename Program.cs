using System;
using System.IO;
using MLNet.Noshow;

namespace MLNet
{
    public class Program
    {
        private static readonly string s_rootPath = Path.Combine(Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]), "..", "..", "..");

        public static void Main()
        {
            var program = new Test(s_rootPath);
            program.Train();
            //program.TestModel();
        }
    }
}
