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

        private readonly string _rootPath;
        private readonly string _modelSavePath;
        private readonly string _dataPath;
        private readonly string _validatePath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _rootPath = rootPath;
            _modelSavePath = GetFilePath("model.zip");
            _dataPath = GetFilePath("data_hmhcks.tsv");
            _validatePath = GetFilePath("data_pmhcks.tsv");
            _context = new MLContext(seed: 0);
        }

        public BinaryClassificationMetrics Evaluate(string modelFileName = "model.zip")
        {
            var testData = GetData(_validatePath);
            var model = _context.Model.Load(GetFilePath(modelFileName), out var _);
            return Evaluate("Saved model", model, testData);
        }

        public void Experiment()
        {
            var data = GetData(_dataPath);
            var validate = GetData(_validatePath);

            var experimentSettings = new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = 30 * 60,
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
                    //UnbalancedSets = true,
                    WeightOfPositiveExamples = 1.6, //new Random().Next(20, 40) / 10,
                    //Sigmoid = 1,
                    CategoricalSmoothing = 1, //Random(0, 1, 10, 20),
                    L2CategoricalRegularization = 1, //Random(0.1, 0.5, 1, 5, 10),
                    MaximumCategoricalSplitPointCount = 16, //Random(8, 16, 32, 64),
                    MinimumExampleCountPerLeaf = 20, //Random(1, 10, 20, 50),
                    MaximumBinCountPerFeature = 200,
                    HandleMissingValue = true,
                    UseZeroAsMissingValue = false,
                    MinimumExampleCountPerGroup = 100, //Random(10, 50, 100, 200),
                    NumberOfIterations = 100,
                    LearningRate = 0.4f, //Random(0.025f, 0.08f, 0.2f, 0.4f),
                    NumberOfLeaves = 128, //Random(2, 16, 64, 128),
                    Booster = new GradientBooster.Options
                    {
                        L1Regularization = 1, //Random(0, 0.5, 1),
                        L2Regularization = 1, //Random(0, 0.5, 1),
                        MaximumTreeDepth = 0,
                        SubsampleFrequency = 0,
                        SubsampleFraction = 1,
                        FeatureFraction = 1,
                        MinimumChildWeight = 0.1,
                        MinimumSplitGain = 0,
                    }
                };

                var trainer = _context.BinaryClassification.Trainers.LightGbm(options);

                var pipeline = CreatePipeline(trainer);

                var model = pipeline.Fit(trainingData);

                var beta = 2;
                var metrics = Evaluate("Test", model, testData, beta);
                //var score = metrics.FBeta(beta);
                var score = (metrics.PositiveRecall + metrics.NegativeRecall) / 2;

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

            static T Random<T>(params T[] choices)
            {
                return choices[new Random().Next(0, choices.Length - 1)];
            }
        }

        public void Predict()
        {
            var model = _context.Model.Load(_modelSavePath, out var schema);

            // NOTE: Use PredictionEnginePool service in real world:
            // https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/serve-model-web-api-ml-net
            var engine = _context.Model.CreatePredictionEngine<Appointment, NoShowPrediction>(model, schema);

            Predict(engine, "1", new Appointment
            {
                NoShow = true,
                LeadTime = 20,
                DayOfWeek = 3,
                Hour = 0,
                Age = 10,
                PreviousNoShows = 5,
                TotalScheduled = 70,
            });

            Predict(engine, "2", new Appointment
            {
                NoShow = false,
                LeadTime = 8,
                DayOfWeek = 5,
                Hour = 0,
                Age = 35,
                PreviousNoShows = 3,
                TotalScheduled = 13,
            });

            Predict(engine, "3", new Appointment
            {
                NoShow = false,
                LeadTime = 2,
                DayOfWeek = 3,
                Hour = 13,
                Age = 29,
                PreviousNoShows = 0,
                TotalScheduled = 3004,
            });
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

        private void Predict(PredictionEngine<Appointment, NoShowPrediction> predictionEngine, string description, Appointment sample)
        {
            var prediction = predictionEngine.Predict(sample);

            Console.WriteLine($"Sample: {description,-20} Predicted: {prediction.NoShow,-5} Actual: {sample.NoShow,-5} Probability: {prediction.Probability,-10:P2}");
        }

        private IDataView GetData(string path)
        {
            return _context.Data.LoadFromTextFile<Appointment>(path, new TextLoader.Options
            {
                HasHeader = true,
            });
        }

        private string GetFilePath(string filename)
        {
            return Path.Combine(_rootPath, "noshowv3", filename);
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
