using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace MLNet.NoshowV3
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.Age),
            nameof(Appointment.LeadTime),
            nameof(Appointment.PreviousNoShows),
            nameof(Appointment.TotalScheduled),
            nameof(Appointment.Hour),
        };

        private static readonly IList<string> s_binnedColumns = new string[]
        {
            nameof(Appointment.Age),
        };

        private static readonly IList<string> s_categoryColumns = new string[]
        {
            nameof(Appointment.DayOfWeek),
        };

        private static readonly string[] s_allFeatureNames = s_columns
            .Select(name => s_categoryColumns.Contains(name)
                ? name + "Encoded" 
                : s_binnedColumns.Contains(name) ? name + "Binned" : name)
            .ToArray();

        private readonly string _modelSavePath;
        private readonly string _dataPath;
        private readonly string _validatePath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "noshowv3", "model.zip");
            _dataPath = Path.Combine(rootPath, "noshowv3", "data_hmhcks.tsv");
            _validatePath = Path.Combine(rootPath, "noshowv3", "data_pmhcks.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Experiment()
        {
            var data = GetData(_dataPath);
            var validate = GetData(_validatePath);

            var experimentSettings = new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = 5 * 60,
                OptimizingMetric = BinaryClassificationMetric.F1Score,
            };

            experimentSettings.Trainers.Clear();
            experimentSettings.Trainers.Add(BinaryClassificationTrainer.AveragedPerceptron);
            experimentSettings.Trainers.Add(BinaryClassificationTrainer.LightGbm);

            var experiment = _context.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var experimentResult = experiment.Execute(
                trainData: data,
                validationData: validate,
                //columnInformation: new ColumnInformation
                //{
                //    ExampleWeightColumnName = nameof(Appointment.Weight)
                //},
                progressHandler: new ProgressHandler());

            Console.WriteLine("Experiment completed");
            Console.WriteLine();

            ConsoleHelper.Print(experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics);

            SaveModel(data.Schema, experimentResult.BestRun.Model);
            Console.WriteLine("Best model saved");
        }

        public void Train()
        {
            var trainingData = GetData(_dataPath);
            var testData = GetData(_validatePath);

            double? bestScore = null;
            while (true)
            {
                var options = new LightGbmBinaryTrainer.Options
                {
                    //ExampleWeightColumnName = nameof(Appointment.Weight),
                    EvaluationMetric = LightGbmBinaryTrainer.Options.EvaluateMetricType.Logloss,
                    UnbalancedSets = true,
                    //WeightOfPositiveExamples = 9,
                    Seed = 459933621,
                    Sigmoid = 1,
                    CategoricalSmoothing = 10,
                    L2CategoricalRegularization = 10,
                    MaximumCategoricalSplitPointCount = 8,
                    MinimumExampleCountPerLeaf = 1,
                    MaximumBinCountPerFeature = 200,
                    HandleMissingValue = true,
                    UseZeroAsMissingValue = false,
                    MinimumExampleCountPerGroup = 100,
                    NumberOfIterations = 100,
                    LearningRate = 0.025,
                    NumberOfLeaves = 64,
                    Booster = new GradientBooster.Options
                    {
                        L1Regularization = 0,
                        L2Regularization = 0.5,
                        MaximumTreeDepth = 0,
                        SubsampleFrequency = 0,
                        SubsampleFraction = 0.5,
                        FeatureFraction = 1,
                        MinimumChildWeight = 0.1,
                        MinimumSplitGain = 0,
                    }
                };

                var trainer = _context.BinaryClassification.Trainers.LightGbm(options);

                var pipeline = CreatePipeline(trainer);

                var model = pipeline.Fit(trainingData);

                var beta = 0.5;
                var score = Evaluate("Test", model, testData, beta).F1Score; // .FBeta(beta);

                if (!bestScore.HasValue || score > bestScore.Value)
                {
                    bestScore = score;
                    SaveModel(trainingData.Schema, model);
                    Console.WriteLine($"Saved new model at {bestScore.Value:P2}");
                }
                else if (bestScore.HasValue)
                {
                    Console.WriteLine($"Best model is still {bestScore.Value:P2}");
                }
            }
        }

        private IEstimator<ITransformer> CreatePipeline()
        {
            var transforms = _context.Transforms;

            var binnedColumns = s_binnedColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name + "Binned", name)).ToArray();
            var categoryColumns = s_categoryColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();

            IEstimator<ITransformer> pipeline = transforms.NormalizeBinning(binnedColumns);

            if (categoryColumns.Length > 0)
                pipeline = pipeline.Append(transforms.Categorical.OneHotEncoding(categoryColumns));

            return pipeline

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames))

                .AppendCacheCheckpoint(_context);
        }

        private EstimatorChain<ISingleFeaturePredictionTransformer<T>> CreatePipeline<T>(ITrainerEstimator<ISingleFeaturePredictionTransformer<T>, T> trainer)
            where T : class
        {
            return CreatePipeline().Append(trainer);
        }

        public BinaryClassificationMetrics Evaluate()
        {
            var testData = GetData(_validatePath);
            var model = _context.Model.Load(_modelSavePath, out var _);
            return Evaluate("Saved model", model, testData);
        }

        private BinaryClassificationMetrics Evaluate(string modelName, ITransformer model, IDataView testData, double beta = 0.5)
        {
            var predictions = model.Transform(testData);
            var metrics = _context.BinaryClassification.EvaluateNonCalibrated(predictions);
            ConsoleHelper.Print(modelName, metrics, beta);
            return metrics;
        }

        private void SaveModel(DataViewSchema schema, ITransformer model)
        {
            _context.Model.Save(model, schema, _modelSavePath);
        }

        private IDataView GetData(string path)
        {
            return _context.Data.LoadFromTextFile<Appointment>(path, new TextLoader.Options
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
