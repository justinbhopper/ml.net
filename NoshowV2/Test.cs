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
using Newtonsoft.Json;

namespace MLNet.NoshowV2
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.Age),
            //nameof(Appointment.DayOfWeek),
            //nameof(Appointment.HasEmergencyContact),
            //nameof(Appointment.Hour),
            //nameof(Appointment.IsFirstInRecurrence),
            //nameof(Appointment.IsRecurring),
            //nameof(Appointment.LastAppointmentNoShow),
            //nameof(Appointment.LastAppointmentScripts),
            nameof(Appointment.LeadTime),
            //nameof(Appointment.Male),
            //nameof(Appointment.Minutes),
            //nameof(Appointment.Month),
            nameof(Appointment.NoShowRatio),
            //nameof(Appointment.OMBAmericanIndian),
            //nameof(Appointment.OMBAsian),
            //nameof(Appointment.OMBBlack),
            //nameof(Appointment.OMBHawaiian),
            //nameof(Appointment.OMBWhite),
            //nameof(Appointment.PreviousNoShows),
            nameof(Appointment.TotalScheduled),
            //nameof(Appointment.Week),
        };

        private static readonly IList<string> s_binnedColumns = new string[]
        {
            nameof(Appointment.Age),
        };

        private static readonly IList<string> s_categoryColumns = new string[]
        {
            nameof(Appointment.DayOfWeek),
        };

        private static readonly IList<string> s_vectorColumns = new string[]
        {
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
            .Select(name => (s_boolColumns.Contains(name) || s_categoryColumns.Contains(name))
                ? name + "Encoded" 
                : s_binnedColumns.Contains(name) ? name + "Binned" : name)
            .ToArray();

        private readonly string _modelSavePath;
        private readonly string _dataPath;
        private readonly string _validatePath;

        private readonly MLContext _context;

        public Test(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "noshowv2", "model.zip");
            _dataPath = Path.Combine(rootPath, "noshowv2", "data_hmhcks_weighted.tsv");
            _validatePath = Path.Combine(rootPath, "noshowv2", "data_pmhcks_weighted.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Experiment()
        {
            var data = GetData(_dataPath);
            var validate = GetData(_validatePath);

            var experimentSettings = new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = 45 * 60,
                OptimizingMetric = BinaryClassificationMetric.F1Score,
            };

            experimentSettings.Trainers.Clear();
            experimentSettings.Trainers.Add(BinaryClassificationTrainer.LightGbm);

            var experiment = _context.Auto().CreateBinaryClassificationExperiment(experimentSettings);

            var experimentResult = experiment.Execute(
                trainData: data,
                validationData: validate,
                columnInformation: new ColumnInformation
                {
                    ExampleWeightColumnName = nameof(Appointment.Weight)
                },
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
                    ExampleWeightColumnName = nameof(Appointment.Weight),
                    EvaluationMetric = LightGbmBinaryTrainer.Options.EvaluateMetricType.Logloss,
                    Sigmoid = 1,
                    CategoricalSmoothing = 10,
                    L2CategoricalRegularization = 10,
                    MaximumCategoricalSplitPointCount = 8,
                    MinimumExampleCountPerLeaf = 1,
                    WeightOfPositiveExamples = 2,
                    MaximumBinCountPerFeature = 200,
                    Seed = 459933621,
                    HandleMissingValue = true,
                    UseZeroAsMissingValue = false,
                    MinimumExampleCountPerGroup = 100,
                    NumberOfIterations = 200,
                    LearningRate = 0.01,
                    NumberOfLeaves = 110,
                    Booster = new GradientBooster.Options
                    {
                        L1Regularization = 0,
                        L2Regularization = 0,
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

                var f1 = Evaluate("Test", model, testData);

                if (!bestScore.HasValue || f1 > bestScore.Value)
                {
                    bestScore = f1;
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

            var boolColumns = s_boolColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();
            var vectorColumns = s_vectorColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name, name)).ToArray();
            var binnedColumns = s_binnedColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name + "Binned", name)).ToArray();
            var categoryColumns = s_categoryColumns.Where(s_columns.Contains).Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();

            IEstimator<ITransformer> pipeline = transforms.NormalizeBinning(binnedColumns);

            if (vectorColumns.Length > 0)
            {
                pipeline = pipeline
                    .Append(transforms.Conversion.MapValueToKey(vectorColumns))
                    .Append(transforms.Conversion.MapKeyToVector(vectorColumns));
            }

            if (categoryColumns.Length > 0)
                pipeline = pipeline.Append(transforms.Categorical.OneHotEncoding(categoryColumns));

            if (boolColumns.Length > 0)
                pipeline = pipeline.Append(transforms.Conversion.ConvertType(boolColumns, DataKind.Single));

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

        public void Evaluate()
        {
            var data = GetData(_dataPath);
            Evaluate(data);
        }

        public void Predict()
        {
            var model = _context.Model.Load(_modelSavePath, out var schema);
            var engine = _context.Model.CreatePredictionEngine<Appointment, NoShowPrediction>(model, schema);
            Predict(engine, "No-show 1", new Appointment
            {
                NoShow = true,
                LeadTime = 20,
                DayOfWeek = 3,
                Month = 4,
                Week = 17,
                Hour = 0,
                Minutes = 60,
                IsRecurring = true,
                IsFirstInRecurrence = false,
                Age = 10,
                Male = true,
                OMBWhite = false,
                OMBAmericanIndian = false,
                OMBAsian = false,
                OMBBlack = false,
                OMBHawaiian = false,
                HasEmergencyContact = true,
                LastAppointmentNoShow = false,
                PreviousNoShows = 5,
                TotalScheduled = 70,
                NoShowRatio = 0.07142857f,
                LastAppointmentScripts = 2,
            });

            Predict(engine, "Show 1", new Appointment
            {
                NoShow = false,
                LeadTime = 8,
                DayOfWeek = 5,
                Month = 4,
                Week = 15,
                Hour = 0,
                Minutes = 60,
                IsRecurring = true,
                IsFirstInRecurrence = false,
                Age = 35,
                Male = true,
                OMBWhite = false,
                OMBAmericanIndian = false,
                OMBAsian = false,
                OMBBlack = false,
                OMBHawaiian = false,
                HasEmergencyContact = true,
                LastAppointmentNoShow = false,
                PreviousNoShows = 3,
                TotalScheduled = 13,
                NoShowRatio = 0.23076923f,
                LastAppointmentScripts = 0,
            });
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

            Console.WriteLine($"Sample: {description,-20} Predicted: {prediction.NoShow,-5} Actual: {sample.NoShow,-5} Probability: {prediction.Probability,-10:P2}");
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
                var model = value.Model;
                ConsoleHelper.Print(value.TrainerName, value.ValidationMetrics);
            }
        }
    }
}
