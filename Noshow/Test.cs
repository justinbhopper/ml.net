using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
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
            _dataPath = Path.Combine(rootPath, "noshow", "data.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Train()
        {
            var data = GetData();

            // Filter out cancelled appointments
            data = _context.Data.FilterRowsByColumn(data, "ShowNoShow", lowerBound: 1, upperBound: 3); 

            var split = _context.Data.TrainTestSplit(data, testFraction: 0.05);
            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            CreateModel(trainingData);
            
            Evaluate(testData);
        }

        private void CreateModel(IDataView trainingData)
        {
            var transforms = _context.Transforms;

            var encodedColumns = s_categoryColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();
            
            var dataProcessPipeline = transforms.CustomMapping(new AppointmentFactory().GetMapping(), contractName: "Appointment")
                .Append(transforms.CopyColumns("Label", nameof(Appointment.NoShow)))
                
                // Put age into separate bins
                .Append(transforms.NormalizeBinning("AgeBinned", nameof(Appointment.Age), maximumBinCount: 10))
                
                //.Append(transforms.Categorical.OneHotEncoding(encodedColumns, OneHotEncodingEstimator.OutputKind.Indicator))

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames));

            var trainer = _context.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                L1Regularization = 0.005f,
            });

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingData);

            var trainsformedData = trainedModel.Transform(trainingData);
            var contributionMetrics = _context.BinaryClassification.PermutationFeatureImportance(trainedModel.LastTransformer, trainsformedData, "Label", numberOfExamplesToUse: 100);
            
            ConsoleHelper.Print(s_allFeatureNames, contributionMetrics, trainsformedData);

            // Save the model
            SaveModel(trainingData.Schema, trainedModel);
        }

        private void Evaluate(IDataView testData)
        {
            var model = _context.Model.Load(_modelSavePath, out var _);
            var predictions = model.Transform(testData);
            var metrics = _context.BinaryClassification.Evaluate(predictions, "Label");
            ConsoleHelper.Print("Test Data", metrics);
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
