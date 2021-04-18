using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet
{
    public class Notes
    {
        private readonly string _modelSavePath;
        private readonly string _dataPath;

        private readonly MLContext _context;

        public Notes(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "notes", "note_model.zip");
            _dataPath = Path.Combine(rootPath, "notes", "notes.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Train()
        {
            var data = GetData();
            var split = _context.Data.TrainTestSplit(data, testFraction: 0.1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var pipeline = ProcessData();

            var model = BuildAndSaveModel(trainData, pipeline);

            Evaluate(model, testData);
        }

        public void TestModel()
        {
            // Load model from disk
            var model = _context.Model.Load(_modelSavePath, out var _);

            Predict(model);
        }

        private IEstimator<ITransformer> ProcessData()
        {
            var transforms = _context.Transforms;

            return transforms.Categorical.OneHotEncoding("ReasonEncoded", "Reason")
                .Append(transforms.Categorical.OneHotEncoding("AgencyEncoded", "Agency"))
                .Append(transforms.Categorical.OneHotEncoding("LocationEncoded", "Location"))
                .Append(transforms.Conversion.MapValueToKey("Label", "NoteType"))
                .Append(transforms.Concatenate("Features", "ReasonEncoded", "AgencyEncoded", "LocationEncoded"))
                .AppendCacheCheckpoint(_context);
        }

        private ITransformer BuildAndSaveModel(IDataView trainingData, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_context.MulticlassClassification.Trainers.LightGbm("Label", "Features"))
                .Append(_context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = trainingPipeline.Fit(trainingData);

            _context.Model.Save(model, trainingData.Schema, _modelSavePath);

            return model;
        }

        private void Evaluate(ITransformer model, IDataView testData)
        {
            var metrics = _context.MulticlassClassification.Evaluate(model.Transform(testData));
            Print(metrics);
        }

        private void Predict(ITransformer model)
        {
            var predictionEngine = _context.Model.CreatePredictionEngine<Note, NotePrediction>(model);

            var note = new Note
            {
                Location = "a",
                Agency = "three",
                Reason = "cough"
            };

            var prediction = predictionEngine.Predict(note);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.NoteType} ===============");
        }

        private void Print(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {metrics.LogLoss:#.###} (0 is best)");
            Console.WriteLine($"*       LogLossReduction: {metrics.LogLossReduction:#.###} (1 is best)");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private IDataView GetData() => _context.Data.LoadFromTextFile<Note>(_dataPath, hasHeader: true);
    }
}
