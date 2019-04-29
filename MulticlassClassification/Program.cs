using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace MulticlassClassification
{
    public class Program
    {
        private static string TrainDataPath = GetAbsolutePath("./Data/TrainData.txt");
        private static string TestDataPath = GetAbsolutePath("./Data/TestData.txt");
        private static string BaseModelPath = "./MLModels";
        private static string ModelPath = GetAbsolutePath($"{BaseModelPath}/TestClassificationModel.zip");

        public static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            var trainingDataView = context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            var testDataView = context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);

            var model = new ModelBuilder();

            var pipeline = model.TrainingModelSetup(context);
            var stopwatch = Stopwatch.StartNew();

            var trainedModel = pipeline.Fit(trainingDataView);
            stopwatch.Stop();
            long elapsedMs = stopwatch.ElapsedMilliseconds;
            Console.WriteLine($"***** Training time: {elapsedMs / 1000} seconds *****");

            var predictions = trainedModel.Transform(trainingDataView);
            var metrics = context.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            context.Model.Save(trainedModel,
                               trainingDataView.Schema,
                               ModelPath);
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
