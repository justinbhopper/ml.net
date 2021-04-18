using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MLNet.Noshow
{
    public class Test
    {
        private static readonly IList<string> s_columns = new[]
        {
            nameof(Appointment.Weekend),
            nameof(Appointment.Season),
            nameof(Appointment.Month),
            nameof(Appointment.Week),
            nameof(Appointment.Hour),
            nameof(Appointment.Sex),
            nameof(Appointment.Age),
            nameof(Appointment.CDCCode),
            nameof(Appointment.CreatedDaysAhead),
            nameof(Appointment.OMBAmericanIndian),
            nameof(Appointment.OMBAsian),
            nameof(Appointment.OMBBlack),
            nameof(Appointment.OMBHawaiian),
            nameof(Appointment.OMBWhite),
            nameof(Appointment.HasEmergencyContact),
        };

        private static readonly IList<string> s_categoryColumns = new[]
        {
            nameof(Appointment.Sex),
            nameof(Appointment.Weekend),
            nameof(Appointment.CDCCode),
            nameof(Appointment.OMBAmericanIndian),
            nameof(Appointment.OMBAsian),
            nameof(Appointment.OMBBlack),
            nameof(Appointment.OMBHawaiian),
            nameof(Appointment.OMBWhite),
            nameof(Appointment.HasEmergencyContact),
        };

        private static readonly string[] s_allFeatureNames = s_columns.Select(name => s_categoryColumns.Contains(name) ? name + "Encoded" : name).ToArray();

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
            var split = _context.Data.TrainTestSplit(data, testFraction: 0.5);
            var trainingData = split.TrainSet;
            var testData = split.TestSet;

            var model = CreateModel(trainingData);
            
            SaveModel(trainingData.Schema, model);

            Evaluate(model, testData);
        }

        private ITransformer CreateModel(IDataView trainingData)
        {
            var transforms = _context.Transforms;

            var encodedColumns = s_categoryColumns.Select(name => new InputOutputColumnPair(name + "Encoded", name)).ToArray();

            var dataProcessPipeline = transforms.CustomMapping<AppointmentInput, Appointment>((src, dest) => dest.Map(src), contractName: "Appointment")
                .Append(transforms.CopyColumns("Label", nameof(Appointment.NoShow)))

                .Append(transforms.Categorical.OneHotEncoding(encodedColumns, OneHotEncodingEstimator.OutputKind.Indicator))

                // Combine data into Features
                .Append(transforms.Concatenate("Features", s_allFeatureNames));

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

            Console.Write($"Sample: {description,-20} Result: {prediction.NoShow,-10} Probability: {prediction.Probability,-10}");
            Console.WriteLine();
        }

        private static void Print(IList<BinaryClassificationMetricsStatistics> permutationMetrics, IDataView transformedData)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Contribution Metrics");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            
            var mapFields = new List<string>();
            foreach (var column in s_allFeatureNames)
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

        private IDataView GetData()
        {
            return _context.Data.LoadFromTextFile<AppointmentInput>(_dataPath, new TextLoader.Options
            {
                HasHeader = true,
            });
        }
    }
}
