using System.Linq;
using MulticlassClassification.Repositories;

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
            
            var classifier = new ClassificationModel(provider);
            var prediction = new IrisDataPrediction(dataRepository);

            classifier.FitAndSaveModel();
            prediction.Probabilities = classifier.PredictValues(prediction.Data, prediction.Categories);
            var predictedCategories = prediction.MaxProbability(prediction.Probabilities);
         
        }
    }
}
