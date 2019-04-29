using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Data;
using MulticlassClassification.DataStructures;

namespace MulticlassClassification
{
    public class Program
    {
        private static string TrainDataPath = GetAbsolutePath(".\\Data\\TrainData.txt");
        private static string TestDataPath = GetAbsolutePath(".\\Data\\TestData.txt");
        private static string BaseModelPath = ".\\MLModels";
        
        private static string ModelPath = GetAbsolutePath($"{BaseModelPath}\\TestClassificationModel.zip");

        public static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            var trainingDataView = context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            var testDataView = context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);
            
            Directory.CreateDirectory(".\\MLModels");
            if (!File.Exists(".\\MLModels\\TestClassificationModel.zip"))
            {
                ZipFile.CreateFromDirectory(".\\MLModels", ".\\MLModels\\TestClassificationModel.zip");
            }
            ZipFile.ExtractToDirectory(".\\MLModels\\TestClassificationModel.zip", ".\\MLModels");

            var model = new ModelBuilder();
            var trainer = model.Trainer(context);

            var pipeline = model.TrainingModelSetup(context);
            var stopwatch = Stopwatch.StartNew();

            var trainedModel = pipeline.Fit(trainingDataView);
            stopwatch.Stop();
            long elapsedMs = stopwatch.ElapsedMilliseconds;
            Console.WriteLine($"***** Training time: {elapsedMs / 1000} seconds *****");

            var predictions = trainedModel.Transform(trainingDataView);
            var metrics = context.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            context.Model.Save(trainedModel,
                               trainingDataView.Schema,
                               ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);
            
            
            var trainedMulticlassModel = context.Model.Load(ModelPath, out var modelInputSchema);
            var predEngine = context.Model.CreatePredictionEngine<IrisData, IrisPrediction>(trainedMulticlassModel);
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

            var IrisFlowers = new Dictionary<float, string>();
            IrisFlowers.Add(0, "Setosa");
            IrisFlowers.Add(1, "Versicolor");
            IrisFlowers.Add(2, "Virginica");
            
            Console.WriteLine("=====Predicting using model====");
            var resultPrediction1 = predEngine.Predict(SampleIrisData.Iris1);
            
            Console.WriteLine($"Actual: Setosa.\n" +
                              $"Predicted label and score:\n" +
                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction1.Score[0]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction1.Score[1]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction1.Score[2]:0.####}\n");
            
            var resultPrediction2 = predEngine.Predict(SampleIrisData.Iris2);

            Console.WriteLine($"Actual: Virginica.\n" +
                              $"Predicted label and score:\n" +
                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction2.Score[0]:0####}\n" +
                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction2.Score[1]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction2.Score[2]:0.####}\n");
            
            var resultPrediction3 = predEngine.Predict(SampleIrisData.Iris3);

            Console.WriteLine($"Actual: Versicolor.\n" +
                              $"Predicted label and score:\n" +
                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction3.Score[0]:0####}\n" +
                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction3.Score[1]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction3.Score[2]:0.####}\n");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
