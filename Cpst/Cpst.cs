using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet
{
    public class Cpst
    {
        private readonly string _modelSavePath;
        private readonly string _dataPath;

        private readonly MLContext _context;

        public Cpst(string rootPath)
        {
            _modelSavePath = Path.Combine(rootPath, "cpst", "model.zip");
            _dataPath = Path.Combine(rootPath, "cpst", "cpst_train.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Train()
        {
            var data = GetData();
            var split = _context.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var estimator = ProcessData();

            var model = BuildAndSaveModel(trainData, estimator);

            Evaluate(model, testData);
        }

        public void TestModel()
        {
            // Load model from disk
            var model = _context.Model.Load(_modelSavePath, out var _);

            Predict(model, "The patient is planning on having a good weekend with staff.", true);
            Predict(model, "The patient is planning on having a good weekend.", false);
            Predict(model, "I discussed with staff about the development of the annual comprehensive assessment.", true);
            Predict(model, "I will develop a comprehensive plan.", false);
        }

        private IEstimator<ITransformer> ProcessData()
        {
            var transforms = _context.Transforms;

            return transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
                .Append(_context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));
        }

        private ITransformer BuildAndSaveModel(IDataView trainingData, IEstimator<ITransformer> estimator)
        {
            var model = estimator.Fit(trainingData);

            _context.Model.Save(model, trainingData.Schema, _modelSavePath);

            return model;
        }

        private void Evaluate(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _context.BinaryClassification.Evaluate(predictions, "Label");
            Print(metrics);
        }

        private void Predict(ITransformer model, string text, bool correctAnswer)
        {
            var predictionEngine = _context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var sample = new SentimentData
            {
                SentimentText = text
            };

            var prediction = predictionEngine.Predict(sample);

            Console.WriteLine($"=============== Single Prediction just-trained-model ===============");
            Console.WriteLine($"Text: {text}");
            Console.WriteLine($"Result: {prediction.Prediction}, Correct Answer: {correctAnswer}, Probability: {prediction.Probability}");
            Console.WriteLine();
        }

        private void Print(CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Auc:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2} (0 is best)");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }

        private IDataView GetData() => _context.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
    }
}
