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

namespace MLNet.NoshowV2
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.Age),
            nameof(Appointment.HasEmergencyContact),
            nameof(Appointment.IsFirstInRecurrence),
            nameof(Appointment.IsRecurring),
            nameof(Appointment.LastAppointmentNoShow),
            nameof(Appointment.LastAppointmentScripts),
            nameof(Appointment.LeadTime),
            nameof(Appointment.Male),
            nameof(Appointment.Minutes),
            nameof(Appointment.NoShowRatio),
            nameof(Appointment.OMBAmericanIndian),
            nameof(Appointment.OMBAsian),
            nameof(Appointment.OMBBlack),
            nameof(Appointment.OMBHawaiian),
            nameof(Appointment.OMBWhite),
            nameof(Appointment.PreviousNoShows),
            nameof(Appointment.TotalScheduled),
        };

        private static readonly IList<string> s_boolColumns = new string[]
        {
            nameof(Appointment.HasEmergencyContact),
            nameof(Appointment.IsFirstInRecurrence),
            nameof(Appointment.IsRecurring),
            nameof(Appointment.LastAppointmentNoShow),
            nameof(Appointment.Male),
            nameof(Appointment.OMBAmericanIndian),
            nameof(Appointment.OMBAsian),
            nameof(Appointment.OMBBlack),
            nameof(Appointment.OMBHawaiian),
            nameof(Appointment.OMBWhite),
        };

        private static readonly string[] s_allFeatureNames = s_columns
            .Select(name => s_boolColumns.Contains(name) ? name + "Encoded" : name)
            .ToArray();

        private readonly string _modelSavePath;
        private readonly string _dataPath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "noshowv2", "model.zip");
            _dataPath = Path.Combine(rootPath, "noshowv2", "data_hmhcks.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Experiment()
        {
            var data = GetData();

            var split = _context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);

            var experimentSettings = new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = 45 * 60,
                OptimizingMetric = BinaryClassificationMetric.F1Score,
            };
            
            experimentSettings.Trainers.Clear();
            experimentSettings.Trainers.Add(BinaryClassificationTrainer.LightGbm);

            var experiment = _context.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var experimentResult = experiment.Execute(
                trainData: split.TrainSet,
                validationData: split.TestSet,
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
                    EvaluationMetric = LightGbmBinaryTrainer.Options.EvaluateMetricType.Logloss,
                    Sigmoid = 0.5,
                    CategoricalSmoothing = 10,
                    L2CategoricalRegularization = 0.1,
                    MaximumCategoricalSplitPointCount = 8,
                    MinimumExampleCountPerLeaf = 20,
                    WeightOfPositiveExamples = 1,
                    MaximumBinCountPerFeature = 255,
                    Seed = 71756196,
                    Verbose = true,
                    HandleMissingValue = true,
                    UseZeroAsMissingValue = false,
                    MinimumExampleCountPerGroup = 50,
                    UnbalancedSets = true,
                    LearningRate = 0.3669092357158661,
                    NumberOfLeaves = 118,
                    Booster = new GradientBooster.Options
                    {
                        L1Regularization = 0,
                        L2Regularization = 1,
                        MaximumTreeDepth = 0,
                        SubsampleFrequency = 0,
                        SubsampleFraction = 1,
                        FeatureFraction = 1,
                        MinimumChildWeight = 0.1,
                        MinimumSplitGain = 0,
                    }
                });

                var pipeline = CreatePipeline(trainer);

                var model = pipeline.Fit(trainingData);

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

        private IEstimator<ITransformer> CreatePipeline()
        {
            var transforms = _context.Transforms;

            var boolColumns = s_boolColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();

            return transforms.Conversion.ConvertType(boolColumns, DataKind.Single)

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames))

                .AppendCacheCheckpoint(_context);
        }

        private EstimatorChain<ISingleFeaturePredictionTransformer<T>> CreatePipeline<T>(ITrainerEstimator<ISingleFeaturePredictionTransformer<T>, T> trainer)
            where T : class
        {
            return CreatePipeline().Append(trainer);
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
            return _context.Data.LoadFromTextFile<Appointment>(_dataPath, new TextLoader.Options
            {
                HasHeader = true,
            });
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
