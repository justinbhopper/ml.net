using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.TrainCatalogBase;

namespace MLNet
{
    public static class ConsoleHelper
    {
        public static void Print(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string name, AnomalyDetectionMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} Anomaly Detection model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       AUC:           {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"*       AUC:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       AUC recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:           {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision: {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:    {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision: {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:    {metrics.NegativeRecall:P2}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string name, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"*       AUC:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       AUC recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"*       PositivePrecision: {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:    {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision: {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:    {metrics.NegativeRecall:P2}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} multi-class classification model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"*       AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"*       LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"*       LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"*       LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"*       LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string algorithmName, IReadOnlyList<CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###}");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(IEnumerable<string> featureNames, IList<BinaryClassificationMetricsStatistics> permutationMetrics, IDataView transformedData)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Contribution Metrics");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");

            var mapFields = new List<string>();
            foreach (var column in featureNames)
            {
                var name = column.Replace("Encoded", "").Replace("Binned", "");

                var slotField = new VBuffer<ReadOnlyMemory<char>>();
                if (transformedData.Schema[column].HasSlotNames())
                {
                    transformedData.Schema[column].GetSlotNames(ref slotField);
                    for (var j = 0; j < slotField.Length; j++)
                    {
                        var item = slotField.GetItemOrDefault(j);
                        mapFields.Add(name + " " + item);
                    }
                }
                else
                {
                    mapFields.Add(name);
                }
            }

            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                .OrderByDescending(feature => Math.Abs(feature.AreaUnderRocCurve.Mean));

            foreach (var feature in sortedIndices)
            {
                Console.WriteLine($"*       {mapFields[feature.index],-30}|\t{Math.Abs(feature.AreaUnderRocCurve.Mean):F6}");
            }

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(IEnumerable<string> featureNames, IList<RegressionMetricsStatistics> permutationMetrics, IDataView transformedData)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Contribution Metrics");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");

            var mapFields = new List<string>();
            foreach (var column in featureNames)
            {
                var name = column.Replace("Encoded", "").Replace("Binned", "");

                var slotField = new VBuffer<ReadOnlyMemory<char>>();
                if (transformedData.Schema[column].HasSlotNames())
                {
                    transformedData.Schema[column].GetSlotNames(ref slotField);
                    for (var j = 0; j < slotField.Length; j++)
                    {
                        var item = slotField.GetItemOrDefault(j);
                        mapFields.Add(name + " " + item);
                    }
                }
                else
                {
                    mapFields.Add(name);
                }
            }

            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.RSquared })
                .OrderByDescending(feature => Math.Abs(feature.RSquared.Mean));

            foreach (var feature in sortedIndices)
            {
                Console.WriteLine($"*       {mapFields[feature.index],-30}|\t{Math.Abs(feature.RSquared.Mean):F6}");
            }

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        public static void Print(string algorithmName, IReadOnlyList<CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            var average = values.Average();
            var sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            var standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            var confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

        public static void Print(string name, ClusteringMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {name} clustering model");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average Distance: {metrics.AverageDistance}");
            Console.WriteLine($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static void ShowDataViewInConsole(IDataView dataView, int numberOfRows = 4)
        {
            var msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                var lineToPrint = "Row--> ";
                foreach (var column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        public static void PeekDataViewInConsole(IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            var msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            // https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview  
            // and iterate through the returned collection from preview.

            var previewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in previewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                var lineToPrint = "Row--> ";
                foreach (var column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekVectorColumnDataInConsole(string columnName, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            var msg = string.Format("Peek data in DataView: : Show {0} rows with just the '{1}' column", numberOfRows, columnName);
            ConsoleWriteHeader(msg);

            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // Extract the 'Features' column.
            var someColumnData = transformedData.GetColumn<float[]>(columnName)
                                                        .Take(numberOfRows).ToList();

            // print to console the peeked rows
            foreach (var row in someColumnData)
            {
                var concatColumn = string.Empty;
                foreach (var f in row)
                {
                    concatColumn += f.ToString();
                }
                Console.WriteLine(concatColumn);
            }
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }
    }
}
