using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLNet
{
    public class GithubIssues
    {
        private readonly string _prepSavePath;
        private readonly string _modelSavePath;
        private readonly string _trainingPath;
        private readonly string _testPath;

        private readonly MLContext _context;

        public GithubIssues(string rootPath)
        {
            _prepSavePath = Path.Combine(rootPath, "github", "prep.zip");
            _modelSavePath = Path.Combine(rootPath, "github", "model.zip");
            _trainingPath = Path.Combine(rootPath, "github", "issues_train.tsv");
            _testPath = Path.Combine(rootPath, "github", "issues_test.tsv");
            _context = new MLContext(seed: 0);
        }

        public void Train()
        {
            var transforms = _context.Transforms;

            var data = GetTrainData();

            // Preparation
            var dataProcessPipeline = transforms.Categorical.OneHotEncoding("PredictedLabel", "Area")
                .Append(transforms.Text.FeaturizeText("TitleFeaturized", "Title"))
                .Append(transforms.Text.FeaturizeText("DescriptionFeaturized", "Description"))
                // Learning alg only reads from Features column
                .Append(transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .Append(_context.Transforms.NormalizeMinMax("Features", "Features"))
                // Sample Caching the DataView so estimators iterating over the data multiple times, 
                // instead of always reading from file, using the cache might get better performance.
                .AppendCacheCheckpoint(_context);

            // Estimator
            var estimator = _context.Regression.Trainers.LbfgsPoissonRegression("PredictedLabel", "Features");

            ITransformer prepModel = dataProcessPipeline.Fit(data);
            var prepData = prepModel.Transform(data);
            ITransformer trainedModel = estimator.Fit(prepData);

            ITransformer trainedPipe = prepModel.Append(trainedModel);

            // Evalulate trained data
            Evaluate(trainedPipe);

            Test(trainedPipe, prepData.Schema);

            // Save model to disk
            _context.Model.Save(prepModel, data.Schema, _prepSavePath);
            _context.Model.Save(trainedModel, prepData.Schema, _modelSavePath);
        }

        public void Retrain()
        {
            var moreData = GetTestData();

            // Load model from disk
            var prepModel = _context.Model.Load(_prepSavePath, out var prepView);
            var trainedModel = _context.Model.Load(_modelSavePath, out var trainedView);

            var originalModelParameters = ((ISingleFeaturePredictionTransformer<object>)trainedModel).Model as LinearRegressionModelParameters;

            // Retrain model with more data
            var retrainedModel = _context.Regression.Trainers.LbfgsPoissonRegression("AreaEncoded", "Features")
                .Fit(moreData, originalModelParameters);

            Test(retrainedModel, trainedView);
        }

        private void Evaluate(ITransformer model)
        {
            var testData = GetTestData();
            Print(_context.Regression.Evaluate(model.Transform(testData), "AreaEncoded"));
        }

        private void Test(ITransformer model, DataViewSchema schema)
        {
            var predictionEngine = _context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);

            VBuffer<ReadOnlyMemory<char>> labels = default;
            schema["AreaEncoded"].GetSlotNames(ref labels);

            var issue = new GitHubIssue
            {
                Title = "Upgrade the HttpClient capacitor",
                Description = "The source code is getting behind in its fluxing abilities and needs updating."
            };

            var prediction = predictionEngine.Predict(issue);

            ReadOnlyMemory<char> area = default;
            labels.GetItemOrDefault(prediction.AreaEncoded, ref area);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {area} ===============");
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

        private void Print(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       RSquared:    {metrics.RSquared:0.###} (1 is best)");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private IDataView GetTrainData() => _context.Data.LoadFromTextFile<GitHubIssue>(_trainingPath, hasHeader: true);
        private IDataView GetTestData() => _context.Data.LoadFromTextFile<GitHubIssue>(_testPath, hasHeader: true);
    }
}
