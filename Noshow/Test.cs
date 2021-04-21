using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;

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

        public void Experiment()
        {
            var data = GetData();

            var split = _context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);

            var pipeline = CreatePrefeaturizedPipeline();

            var experimentSettings = new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = 45 * 60,
                OptimizingMetric = BinaryClassificationMetric.F1Score,
            };

            var experiment = _context.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var experimentResult = experiment.Execute(
                trainData: split.TrainSet,
                validationData: split.TestSet,
                //preFeaturizer: pipeline,
                labelColumnName: nameof(Appointment.NoShow),
                progressHandler: new ProgressHandler());

            Console.WriteLine("Experiment completed");
            Console.WriteLine();

            ConsoleHelper.Print(experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics);

            SaveModel(data.Schema, experimentResult.BestRun.Model);
            Console.WriteLine("Best model saved");
        }

        public void Train()
        {
            var data = GetData();

            var split = _context.Data.TrainTestSplit(data, testFraction: 0.05, seed: 0);
            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            double? best = null;
            while (true)
            {

                var trainer = _context.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
                {
                    NumberOfThreads = 1, // Use 1 to ensure deterministic results
                    //L1Regularization = 0.005f,
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

        /// <summary>
        /// Undersampling can be used to balance the data better, but at the cost of under-representing
        /// the data that will be encountered in the real world.  Be careful to use it only against training 
        /// data and not your test set.
        /// </summary>
        private IDataView UndersampleShows(IDataView data)
        {
            return _context.Data.FilterByCustomPredicate<AppointmentInput>(data, a =>
            {
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

                return false;
            });
        }

        private IEstimator<ITransformer> CreatePrefeaturizedPipeline()
        {
            var transforms = _context.Transforms;

            return transforms.CopyColumns("Label", nameof(Appointment.NoShow))

                // Put age into separate bins
                .Append(transforms.NormalizeBinning("AgeBinned", nameof(Appointment.Age), maximumBinCount: 10));
        }

        private IEstimator<ITransformer> CreatePipeline()
        {
            var transforms = _context.Transforms;

            var encodedColumns = s_categoryColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();

            return CreatePrefeaturizedPipeline()

                //.Append(transforms.Categorical.OneHotEncoding(encodedColumns, OneHotEncodingEstimator.OutputKind.Indicator))

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames))

                .AppendCacheCheckpoint(_context);
        }

        private EstimatorChain<ISingleFeaturePredictionTransformer<T>> CreatePipeline<T>(ITrainerEstimator<ISingleFeaturePredictionTransformer<T>, T> trainer)
            where T : class
        {
            return CreatePipeline().Append(trainer);
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
            var metrics = _context.BinaryClassification.EvaluateNonCalibrated(predictions);
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
            var data = _context.Data.LoadFromTextFile<AppointmentInput>(_dataPath, new TextLoader.Options
            {
                HasHeader = true,
            });

            // Filter out cancelled appointments
            data = _context.Data.FilterByCustomPredicate<AppointmentInput>(data, a =>
            {
                if (a.ShowNoShow != 1 && a.ShowNoShow != 2)
                    return true;

                if (a.Sex != "M" && a.Sex != "F")
                    return true;

                return false;
            });

            // Transform the data
            var transformer = _context.Transforms.CustomMapping(new AppointmentFactory().GetMapping(), contractName: "Appointment")
                .Append(_context.Transforms.DropColumns(nameof(AppointmentInput.ShowNoShow)));

            return transformer.Fit(data).Transform(data);
        }

        private class ProgressHandler : IProgress<RunDetail<BinaryClassificationMetrics>>
        {
            public void Report(RunDetail<BinaryClassificationMetrics> value)
            {
                ConsoleHelper.Print(value.TrainerName, value.ValidationMetrics);
            }
        }
    }
}
