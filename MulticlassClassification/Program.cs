using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Data;
using MulticlassClassification.DataStructures;
using Microsoft.ML.Transforms;

namespace MulticlassClassification
{
    public class Program
    {
        private static string BaseModelPath = ".\\MLModels";
        private static string ModelZipFilePath = $"{BaseModelPath}\\TestClassificationModel.zip";

        private static string ModelPath = GetAbsolutePath($"{BaseModelPath}\\TestClassificationModel.zip");
        private static string TrainDataPath = GetAbsolutePath(".\\Data\\TrainData.txt");
        private static string TestDataPath = GetAbsolutePath(".\\Data\\TestData.txt");

        public static void Main(string[] args)
        {
            //all of this can be extracted into a Classifier(??) class
            var context = new MLContext(seed: 0);
            var trainingDataView = context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            var testDataView = context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true); 

            var model = new IrisModelBuilder();
            var trainer = model.CreateTrainerForModel(context);
            var pipeline = model.DataPipelineSetup(context).Append(trainer);

            CreateDirectoryAndExtractZipfile(BaseModelPath, ModelZipFilePath);
            FitAndSaveModel(context, trainingDataView, trainer, pipeline);

            var trainedMulticlassModel = context.Model.Load(ModelPath, out var modelInputSchema);
            PredictTestValues(context, trainedMulticlassModel);
            //var dataPredictions = PredictValues(context, trainedMulticlassModel, irisData);
        }

        private static void FitAndSaveModel(MLContext context, 
            IDataView trainingDataView, 
            EstimatorChain<KeyToValueMappingTransformer> trainer, 
            EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> pipeline)
        {
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
        }

        private static void PredictTestValues(MLContext context, ITransformer trainedMulticlassModel)
        {
            var categories = new List<string>()
            {
                "Setosa",
                "Virginica",
                "Versicolor"
            };
            var IrisFlowers = OutputCategories(categories);

            var predEngine = context.Model.CreatePredictionEngine<IrisData, Prediction>(trainedMulticlassModel);
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();
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
                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction2.Score[0]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction2.Score[1]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction2.Score[2]:0.####}\n");

            var resultPrediction3 = predEngine.Predict(SampleIrisData.Iris3);

            Console.WriteLine($"Actual: Versicolor.\n" +
                              $"Predicted label and score:\n" +
                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction3.Score[0]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction3.Score[1]:0.####}\n" +
                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction3.Score[2]:0.####}\n");
        }

        private static IEnumerable<Dictionary<string, float>> PredictValues(MLContext context, ITransformer trainedMulticlassModel, IEnumerable<IrisData> dataToPredict)
        {
            var categories = new List<string>()
            {
                "Setosa",
                "Virginica",
                "Versicolor"
            };
            var IrisFlowers = OutputCategories(categories);

            var predEngine = context.Model.CreatePredictionEngine<IrisData, Prediction>(trainedMulticlassModel);
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

            var probabilities = new List<Dictionary<string, float>>();
            foreach(var data in dataToPredict)
            {
                var resultPrediction = predEngine.Predict(data);
                var probabilitiesByLabel = new Dictionary<string, float>();
                for (int i = 0; i < labelsArray.Length; i++)
                {
                    probabilitiesByLabel.Add(IrisFlowers[labelsArray[i]], resultPrediction.Score[i]);
                }
                probabilities.Add(probabilitiesByLabel);
            }
            return probabilities;
        }

        private static void CreateDirectoryAndExtractZipfile(string dirPath, string zipfileLocation)
        {
            Directory.CreateDirectory(dirPath);
            if (!File.Exists(zipfileLocation))
            {
                ZipFile.CreateFromDirectory(dirPath, zipfileLocation);
            }
            ZipFile.ExtractToDirectory(zipfileLocation, dirPath);
        }

        private static Dictionary<float, string> OutputCategories(List<string> categories)
        {
            var indexToCategory = new Dictionary<float, string>();
            var index = 0;
            foreach(var category in categories)
            {
                indexToCategory.Add(index, category);
                index = index + 1;
            }
            return indexToCategory;
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
