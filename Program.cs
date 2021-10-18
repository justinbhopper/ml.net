﻿using System;
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
            program.Train();

            Console.WriteLine($"Total time took {stopwatch.ElapsedMilliseconds / 1000} seconds.");
        }
    }
}
