using System;
using MulticlassClassification.Iris;
using MulticlassClassification.Iris.Repositories;

namespace MulticlassClassification
{
    public class Program
    {    
        public static void Main(string[] args)
        {
            IIrisDataParser dataParser = new IrisDataParser();
            IDataProvider provider = new IrisDataProvider();
            IIrisDataRepository dataRepository = new IrisDataRepository(dataParser);

            provider.CreateDirectoryAndExtractZipfile(provider.BaseModelPath, provider.ModelZipFilePath);
            
            var classifier = new ClassificationModel.ClassificationModel(provider);
            var prediction = new IrisDataPrediction(dataRepository);

            classifier.FitAndSaveModel();
            prediction.Probabilities = classifier.PredictValues(prediction.Data, prediction.Categories);
            var predictedCategories = prediction.PredictCategory(prediction.Probabilities);
            var predictedIrises = prediction.PredictedData(prediction.Data, predictedCategories);

            foreach (var iris in predictedIrises)
            {
                Console.WriteLine($"Sepal Length: {iris.SepalLength}\n" +
                                  $"Sepal Width: {iris.SepalWidth}\n" +
                                  $"Petal Length: {iris.PetalLength}\n" +
                                  $"Petal Width: {iris.PetalWidth}\n" +
                                  $"Predicted type: {iris.Label}\n" +
                                  $"------------------------------");
            }
        }
    }
}
