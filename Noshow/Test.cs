using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLNet.Noshow
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.LeadTime),
            nameof(Appointment.PreviousNoShows),
            nameof(Appointment.TotalScheduled),
            nameof(Appointment.NoShowRatio),
            nameof(Appointment.Age),
        };

        private static readonly IList<string> s_categoryColumns = new string[]
        {
        };

        private static readonly string[] s_allFeatureNames = s_columns
            .Select(name => s_categoryColumns.Contains(name) ? name + "Encoded" : name)
            .Where(name => name != "Age").Concat(new[] { "AgeBinned" })
            .ToArray();

        private readonly string _modelSavePath;
        private readonly string _dataPath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "noshow", "model.zip");
            _dataPath = Path.Combine(rootPath, "noshow", "data_hmhcks.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Train()
        {
            var data = GetData();

            var split = _context.Data.TrainTestSplit(data, testFraction: 0.05, seed: 0);
            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            // Filter out cancelled appointments
            trainingData = _context.Data.FilterByCustomPredicate<AppointmentInput>(trainingData, a =>
            {
                if (a.ShowNoShow != 1 && a.ShowNoShow != 2)
                    return true;

                if (a.Sex != "M" && a.Sex != "F")
                    return true;

                /*
                // Balance number of shows and no-shows using features we know don't normally matter
                if (a.ShowNoShow == 1)
                {
                    var date = DateTime.Parse(a.AppointmentDate);
                    if (date.DayOfWeek == DayOfWeek.Monday || date.DayOfWeek == DayOfWeek.Thursday || date.DayOfWeek == DayOfWeek.Saturday)
                        return true;
                    if (a.ClientKey.Contains("A") || a.ClientKey.Contains("1") || a.ClientKey.Contains("3") || a.ClientKey.Contains("5") || a.ClientKey.Contains("7"))
                        return true;
                    if (a.Sex == "F" && a.HasEmergencyContact == "1")
                        return true;
                }
                */

                return false;
            });

            double? best = null;
            while (true)
            {
                var trainer = _context.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    NumberOfThreads = 1, // Use 1 to ensure deterministic results
                    L1Regularization = 0.005f,
                });

                var pipeline = CreatePipeline(trainer);

                var model = CreateModel(pipeline, trainingData);

                var f1 = Evaluate("Test", model, testData);

                if (!best.HasValue || f1 > best.Value)
                {
                    best = f1;
                    SaveModel(trainingData.Schema, model);
                    Console.WriteLine($"Saved new model at {best.Value:P2}");
                }
                else if (best.HasValue)
                {
                    Console.WriteLine($"Best model is still {best.Value:P2}");
                }
            }
        }

        private EstimatorChain<ISingleFeaturePredictionTransformer<T>> CreatePipeline<T>(ITrainerEstimator<ISingleFeaturePredictionTransformer<T>, T> trainer)
            where T : class
        {
            var transforms = _context.Transforms;

            var encodedColumns = s_categoryColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();
            
            return transforms.CustomMapping(new AppointmentFactory().GetMapping(), contractName: "Appointment")
                .Append(transforms.CopyColumns("Label", nameof(Appointment.NoShow)))

                // Put age into separate bins
                .Append(transforms.NormalizeBinning("AgeBinned", nameof(Appointment.Age), maximumBinCount: 10))

                //.Append(transforms.Categorical.OneHotEncoding(encodedColumns, OneHotEncodingEstimator.OutputKind.Indicator))

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames))
                
                .AppendCacheCheckpoint(_context)
                
                .Append(trainer);
        }

        private ITransformer CreateModel<T>(EstimatorChain<ISingleFeaturePredictionTransformer<T>> pipeline, IDataView trainingData)
            where T : class
        {
            var trainedModel = pipeline.Fit(trainingData);

            var trainsformedData = trainedModel.Transform(trainingData);
            var contributionMetrics = _context.BinaryClassification.PermutationFeatureImportance(trainedModel.LastTransformer, trainsformedData, "Label", numberOfExamplesToUse: 100);

            ConsoleHelper.Print(s_allFeatureNames, contributionMetrics, trainsformedData);

            return trainedModel;
        }

        public void Evaluate()
        {
            var data = GetData();
            Evaluate(data);
        }

        private void Evaluate(IDataView testData)
        {
            var model = _context.Model.Load(_modelSavePath, out var _);
            Evaluate("Saved model", model, testData);
        }

        private double Evaluate(string modelName, ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _context.BinaryClassification.Evaluate(predictions);
            ConsoleHelper.Print(modelName, metrics);
            return metrics.F1Score;
        }

        private void SaveModel(DataViewSchema schema, ITransformer model)
        {
            _context.Model.Save(model, schema, _modelSavePath);
        }

        private void Predict(PredictionEngine<Appointment, NoShowPrediction> predictionEngine, string description, Appointment sample)
        {
            var prediction = predictionEngine.Predict(sample);

            Console.Write($"Sample: {description,-20} Result: {prediction.NoShow,-10} Probability: {prediction.Probability,-10}");
            Console.WriteLine();
        }

        private IDataView GetData()
        {
            return _context.Data.LoadFromTextFile<AppointmentInput>(_dataPath, new TextLoader.Options
            {
                HasHeader = true,
            });
        }
    }
}
