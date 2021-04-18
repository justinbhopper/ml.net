using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MLNet.SimulatedNoshow
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.Weekend),
            nameof(Appointment.Season),
            nameof(Appointment.Month),
            nameof(Appointment.Week),
            nameof(Appointment.Holiday),
            nameof(Appointment.PriorNoShows),
            nameof(Appointment.Male),
            nameof(Appointment.Age),
            nameof(Appointment.Married),
            nameof(Appointment.Alone),
            nameof(Appointment.Medicated),
            nameof(Appointment.NeedsRefill),
        };

        private static readonly IList<string> s_boolColumns = new[]
        {
            nameof(Appointment.Weekend),
            nameof(Appointment.Holiday),
            nameof(Appointment.Male),
            nameof(Appointment.Married),
            nameof(Appointment.Alone),
            nameof(Appointment.Medicated),
            nameof(Appointment.NeedsRefill),
        };

        private readonly string _modelSavePath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "SimulatedNoshow", "model.zip");
            _context = new MLContext(seed: 0);
        }

        public static bool Rules(Appointment item)
        {
            // Children never no-show
            if (item.Age < 16)
                return false;

            // They don't come if its a holiday unless they live alone unmarried
            if (item.Holiday && !item.Married && item.Alone)
                return true;

            // Married men stop coming in summer
            if (item.Male && item.Married && item.Season == 2)
                return true;

            // Weekend holiday
            if (item.Weekend && item.Holiday)
                return true;

            // Males don't come on the weekend
            if (item.Weekend && item.Male)
                return true;

            // They come for refills
            if (item.NeedsRefill)
                return true;

            // They stop coming when they were medicated  
            if (item.Medicated && !item.NeedsRefill)
                return true;

            // They don't come during this week for whatever reason
            if (item.Week == 50)
                return true;

            return false;
        }

        public void Train()
        {
            var data = GetData(100000, Rules, 0.1);
            var split = _context.Data.TrainTestSplit(data, testFraction: 0.1);
            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            var model = CreateModel(trainingData);
            
            SaveModel(trainingData.Schema, model);

            Evaluate(model, testData);
        }

        public void TestModel()
        {
            // Load model from disk
            var model = _context.Model.Load(_modelSavePath, out var _);
            var predictionEngine = _context.Model.CreatePredictionEngine<Appointment, NoShowPrediction>(model);

            Predict(predictionEngine, "Holiday", Data.Single(appt =>
            {
                if (appt.Age < 16)
                    appt.Age += 20;

                appt.Holiday = true;
            }));

            Predict(predictionEngine, "Weekend male", Data.Single(appt =>
            {
                if (appt.Age < 16)
                    appt.Age += 20;

                appt.Weekend = true;
                appt.Male = true;
            }));

            Predict(predictionEngine, "Needs refill", Data.Single(appt =>
            {
                if (appt.Age < 16)
                    appt.Age += 20;

                appt.NeedsRefill = true;
            }));

            var missed = 0;
            var missedHorriably = 0;
            var totalSamples = 10000;
            for (var i = 0; i < totalSamples; ++i)
            {
                var sample = Data.Single();
                var prediction = predictionEngine.Predict(sample);
                if (Rules(sample) != prediction.NoShow)
                {
                    missed++;

                    if (prediction.NoShow && prediction.Probability > 0.9)
                        missedHorriably++;
                    else if (!prediction.NoShow && prediction.Probability < 0.1)
                        missedHorriably++;
                }
            }

            Console.WriteLine($"Random sampling missed {missed} out of {totalSamples} ({missedHorriably} were way off)");
        }

        private ITransformer CreateModel(IDataView trainingData)
        {
            var transforms = _context.Transforms;

            var encodedColumns = s_boolColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();
            var allFeatureNames = s_columns.Select(name => s_boolColumns.Contains(name) ? name + "Encoded" : name).ToArray();

            var dataProcessPipeline = transforms.CopyColumns("Label", nameof(Appointment.NoShow))
                .Append(transforms.Categorical.OneHotEncoding(encodedColumns, OneHotEncodingEstimator.OutputKind.Indicator))

                // Combine data into Features
                .Append(transforms.Concatenate("Features", allFeatureNames));

            var trainer = _context.BinaryClassification.Trainers.FastTree("Label", "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingData);

            var trainsformedData = trainedModel.Transform(trainingData);
            var contributionMetrics = _context.BinaryClassification.PermutationFeatureImportance(trainedModel.LastTransformer, trainsformedData, "Label", numberOfExamplesToUse: 100);
            
            Print(contributionMetrics, trainsformedData);

            return trainedModel;
        }

        private void SaveModel(DataViewSchema schema, ITransformer model)
        {
            _context.Model.Save(model, schema, _modelSavePath);
        }

        private void Evaluate(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _context.BinaryClassification.Evaluate(predictions, "Label");
            Print(metrics);
        }

        private void Predict(PredictionEngine<Appointment, NoShowPrediction> predictionEngine, string description, Appointment sample)
        {
            var prediction = predictionEngine.Predict(sample);

            Console.Write($"Sample: {description,-20} Result: {prediction.NoShow,-10} Correct Answer: {Rules(sample),-10} Probability: {prediction.Probability,-10}");
            Console.WriteLine();
        }

        private void Print(IList<BinaryClassificationMetricsStatistics> permutationMetrics, IDataView transformedData)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Contribution Metrics");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            
            var mapFields = new List<string>();
            var allFeatureNames = s_columns.Select(name => s_boolColumns.Contains(name) ? name + "Encoded" : name).ToArray();
            foreach (var column in allFeatureNames)
            {
                var name = column.Replace("Encoded", "");

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

        private void Print(CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Auc:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        private IDataView GetData(int count, Func<Appointment, bool> outcome, double variance)
        {
            return _context.Data.LoadFromEnumerable(Data.Train(count, outcome, variance));
        }
    }
}
